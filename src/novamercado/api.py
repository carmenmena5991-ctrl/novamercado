"""API HTTP del servicio. Los endpoints solo orquestan: la limpieza vive en `limpieza`,
el pronóstico en `predictor` y la persistencia en `db`."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .airflow_cliente import ErrorAirflow
from .catalogos import Catalogos
from .esquemas import (EJEMPLO_OBSERVACION, EJEMPLO_PREDICCION, RespuestaHealth,
                       RespuestaModelo, RespuestaObservacion, RespuestaPrediccion,
                       RespuestaReentrenarEstado, RespuestaReentrenarInicio,
                       SolicitudObservacion, SolicitudPrediccion)
from .limpieza import Rechazo, limpiar_observacion, limpiar_prediccion
from .predictor import pronosticar
from .servicio import GestorModelo

log = logging.getLogger(__name__)


@dataclass
class Dependencias:
    catalogos: Catalogos
    repo: Any  # RepositorioPostgres o un doble de pruebas con la misma interfaz
    gestor: GestorModelo
    airflow: Any  # ClienteAirflow o doble
    nombre_modelo: str
    alias: str
    al_iniciar: Callable[[], None] | None = None
    _maximos: dict = field(default_factory=dict)

    def maximos(self) -> dict:
        if not self._maximos:
            try:
                self._maximos = self.repo.maximos_historicos()
            except Exception as e:  # noqa: BLE001
                log.warning("no se pudieron leer máximos históricos: %s", e)
        return self._maximos


def _rechazo(detalle: str, codigo: int = 422) -> JSONResponse:
    return JSONResponse(status_code=codigo, content={"detail": detalle})


def crear_app(dep: Dependencias) -> FastAPI:
    @asynccontextmanager
    async def ciclo_vida(_app: FastAPI):
        if dep.al_iniciar:
            dep.al_iniciar()
        dep.gestor.iniciar()
        yield
        dep.gestor.detener()

    app = FastAPI(title="NovaMercado - Pronóstico diario de ventas", version="1.0.0",
                  lifespan=ciclo_vida)

    def _registrar_rechazo(sid: str, endpoint: str, cuerpo: Any, detalle: str) -> None:
        try:
            dep.repo.registrar_solicitud(sid, endpoint, cuerpo, 422, detalle=detalle)
        except Exception as e:  # noqa: BLE001 - el rechazo se responde aunque no se registre
            log.error("no se pudo registrar el rechazo %s: %s", sid, e)

    # ------------------------------------------------------------------ predicción
    @app.post("/v1/prediccion", response_model=RespuestaPrediccion,
              responses={422: {"description": "Solicitud rechazada"},
                         503: {"description": "Sin modelo o sin base de datos"}})
    def prediccion(cuerpo: Any = Body(..., examples=[EJEMPLO_PREDICCION])):
        sid = str(uuid.uuid4())
        try:
            limpio, bit = limpiar_prediccion(cuerpo, dep.catalogos)
            solicitud = SolicitudPrediccion.model_validate(limpio)
        except Rechazo as r:
            _registrar_rechazo(sid, "/v1/prediccion", cuerpo, r.detalle)
            return _rechazo(r.detalle)
        except ValidationError as e:
            _registrar_rechazo(sid, "/v1/prediccion", cuerpo, str(e))
            return _rechazo(f"la solicitud no cumple el contrato: {e.errors()[0]['msg']}")

        modelo = dep.gestor.actual
        if modelo is None:
            return _rechazo("el servicio aún no tiene un modelo cargado", 503)
        try:
            historial = dep.repo.historial_tienda(solicitud.tienda_id, solicitud.fecha)
        except Exception as e:  # noqa: BLE001
            log.error("base de datos no disponible: %s", e)
            return _rechazo("la base de datos no está disponible", 503)
        try:
            preds = pronosticar(modelo.bundle, limpio, historial, dep.catalogos, bit)
        except Rechazo as r:
            _registrar_rechazo(sid, "/v1/prediccion", cuerpo, r.detalle)
            return _rechazo(r.detalle)

        respuesta = RespuestaPrediccion(
            tienda_id=solicitud.tienda_id, fecha=solicitud.fecha, version_modelo=modelo.version,
            predicciones=[{k: p[k] for k in ("familia_id", "venta_esperada", "cantidad_sugerida")}
                          for p in preds],
            calidad=bit.calidad())
        try:
            dep.repo.registrar_solicitud(sid, "/v1/prediccion", cuerpo, 200, solicitud.tienda_id,
                                         solicitud.fecha, bit.calidad(), version=modelo.version,
                                         predicciones=preds)
        except Exception as e:  # noqa: BLE001
            log.error("no se pudo registrar la predicción %s: %s", sid, e)
        return respuesta

    # ---------------------------------------------------------------- observaciones
    @app.post("/v1/observaciones", response_model=RespuestaObservacion,
              responses={422: {"description": "Observación rechazada"}})
    def observaciones(cuerpo: Any = Body(..., examples=[EJEMPLO_OBSERVACION])):
        sid = str(uuid.uuid4())
        try:
            limpio, bit = limpiar_observacion(cuerpo, dep.catalogos, dep.maximos())
            obs = SolicitudObservacion.model_validate(limpio)
        except Rechazo as r:
            _registrar_rechazo(sid, "/v1/observaciones", cuerpo, r.detalle)
            return _rechazo(r.detalle)
        except ValidationError as e:
            _registrar_rechazo(sid, "/v1/observaciones", cuerpo, str(e))
            return _rechazo(f"la observación no cumple el contrato: {e.errors()[0]['msg']}")
        try:
            n = dep.repo.registrar_observacion(sid, cuerpo, obs.tienda_id, obs.fecha,
                                               limpio["ventas"], bit)
        except Exception as e:  # noqa: BLE001
            log.error("no se pudo registrar la observación: %s", e)
            return _rechazo("la base de datos no está disponible", 503)
        return RespuestaObservacion(registros_guardados=n, calidad=bit.calidad())

    # ------------------------------------------------------------- administración
    @app.post("/admin/reentrenar", status_code=202, response_model=RespuestaReentrenarInicio,
              responses={503: {"description": "No se pudo iniciar"}})
    def reentrenar():
        run_id = f"reentreno_{uuid.uuid4().hex[:12]}"
        try:
            dep.repo.crear_reentrenamiento(run_id)
        except Exception as e:  # noqa: BLE001
            return _rechazo(f"no se pudo registrar la ejecución: {e}", 503)
        try:
            dep.airflow.disparar(run_id)
        except ErrorAirflow as e:
            dep.repo.borrar_reentrenamiento(run_id)
            return _rechazo(f"no se pudo iniciar el reentrenamiento: {e}", 503)
        return {"id": run_id}

    @app.get("/admin/reentrenar/{run_id}", response_model=RespuestaReentrenarEstado,
             response_model_exclude_none=True, responses={404: {"description": "id desconocido"}})
    def estado_reentrenamiento(run_id: str):
        fila = dep.repo.obtener_reentrenamiento(run_id)
        if fila is None:
            return JSONResponse(status_code=404, content={"detail": f"ejecución {run_id} no existe"})
        if fila["estado"] in ("exitoso", "fallido"):
            return {k: fila[k] for k in ("estado", "promovido", "version", "motivo")}
        try:
            estado = dep.airflow.estado(run_id)
        except ErrorAirflow:
            return {"estado": "en_curso"}
        if estado in ("failed", None) or (estado == "success"):
            # El DAG escribe su resultado en su última tarea: si terminó sin escribirlo, falló.
            fila = dep.repo.obtener_reentrenamiento(run_id)
            if fila and fila["estado"] in ("exitoso", "fallido"):
                return {k: fila[k] for k in ("estado", "promovido", "version", "motivo")}
            if estado is None:
                return {"estado": "en_curso"}
            actual = dep.gestor.actual
            version = actual.version if actual else None
            motivo = ("la ejecución falló en Airflow; el modelo vigente no cambió"
                      if estado == "failed" else "la ejecución terminó sin registrar resultado")
            dep.repo.finalizar_reentrenamiento(run_id, "fallido", False, version, motivo)
            return {"estado": "fallido", "promovido": False, "version": version, "motivo": motivo}
        return {"estado": "en_curso"}

    @app.get("/modelo", response_model=RespuestaModelo)
    def modelo_vigente():
        actual = dep.gestor.actual
        if actual is None:
            return _rechazo("no hay modelo cargado", 503)
        return {"nombre": dep.nombre_modelo, "alias": dep.alias, "version": actual.version}

    @app.get("/health", response_model=RespuestaHealth, response_model_exclude_none=True)
    def health():
        actual = dep.gestor.actual
        if actual is None:
            return JSONResponse(status_code=503, content={
                "estado": "degradado", "motivo": "no hay un modelo cargado todavía"})
        problemas = dep.gestor.problemas()
        detalle = {"version_modelo": actual.version}
        if problemas:
            motivo = "; ".join(f"{k} no disponible ({v})" for k, v in problemas.items())
            return {"estado": "degradado", "motivo": motivo, **detalle}
        return {"estado": "ok", **detalle}

    return app
