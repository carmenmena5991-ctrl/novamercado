"""Fixtures: catálogos sintéticos y dobles en memoria (sin MLflow, sin Postgres, sin datos
publicados; los datos no forman parte del repositorio)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from novamercado.airflow_cliente import ErrorAirflow
from novamercado.api import Dependencias, crear_app
from novamercado.catalogos import Catalogos
from novamercado.modelo import crear_bundle
from novamercado.servicio import GestorModelo

FAMILIAS = [("abarrotes", "Abarrotes"), ("bebidas", "Bebidas"), ("lacteos", "Lácteos"),
            ("carnes", "Carnes"), ("panaderia", "Panadería"),
            ("frutas_verduras", "Frutas y verduras"), ("limpieza", "Limpieza"),
            ("cuidado_personal", "Cuidado personal"), ("licores", "Licores"),
            ("snacks", "Snacks y dulces"), ("bebe", "Bebé"), ("mascotas", "Mascotas")]


@pytest.fixture
def cat() -> Catalogos:
    tiendas = pd.DataFrame({
        "tienda_id": range(1, 11),
        "nombre": [f"NovaMercado T{i}" for i in range(1, 10)] + ["NovaMercado Express Zona 16"],
        "ciudad": ["Guatemala"] * 4 + ["Quetzaltenango", "Antigua Guatemala", "Escuintla",
                                       "Cobán", "Chimaltenango", "Guatemala"],
        "tipo": ["hipermercado", "supermercado", "hipermercado", "express", "hipermercado",
                 "supermercado", "supermercado", "supermercado", "express", "express"],
        "fecha_apertura": ["2019-01-01"] * 9 + ["2025-03-15"],
    }).set_index("tienda_id")
    familias = pd.DataFrame(FAMILIAS, columns=["familia_id", "nombre"]).set_index("familia_id")
    costos = pd.DataFrame({"familia_id": [f for f, _ in FAMILIAS],
                           "costo_faltante": [1.0] * 12,
                           "costo_sobrante": [0.2] * 12}).set_index("familia_id")
    costos.loc["panaderia"] = [1.0, 2.5]
    feriados = pd.DataFrame({
        "fecha": ["2024-03-28", "2025-11-01", "2025-12-25", "2026-12-25", "2026-08-04"],
        "nombre": ["Jueves Santo", "Todos los Santos", "Navidad", "Navidad", "Santo Domingo"],
        "tipo": ["feriado"] * 5,
        "ambito": ["nacional"] * 4 + ["Cobán"],
    })
    return Catalogos(tiendas, familias, costos, feriados)


def solicitud_valida(**cambios) -> dict:
    base = {"tienda_id": 5, "fecha": "2025-11-17",
            "familias": [{"familia_id": f, "en_promocion": 0} for f, _ in FAMILIAS]}
    base.update(cambios)
    return base


def observacion_valida(**cambios) -> dict:
    base = {"tienda_id": 5, "fecha": "2025-11-17",
            "ventas": [{"familia_id": f, "ventas": 100 + i} for i, (f, _) in enumerate(FAMILIAS)]}
    base.update(cambios)
    return base


class RepoMemoria:
    """Doble de RepositorioPostgres con la misma interfaz pública usada por la API."""

    def __init__(self):
        filas = []
        for t in range(1, 11):
            for d in pd.date_range("2025-09-01", "2025-11-16"):
                for i, (f, _) in enumerate(FAMILIAS):
                    filas.append((t, f, d.date(), float(500 + 100 * i + d.weekday()), 0))
        self.historico = pd.DataFrame(filas, columns=["tienda_id", "familia_id", "fecha",
                                                      "ventas", "en_promocion"])
        self.solicitudes: list[dict] = []
        self.observaciones: dict[tuple, float] = {}
        self.reentrenos: dict[str, dict] = {}

    def ping(self):
        pass

    def historial_tienda(self, tienda_id, fecha, dias=60):
        h = self.historico
        h = h[(h["tienda_id"] == tienda_id) & (h["fecha"] < fecha)]
        if h.empty:
            return h
        return h[h["fecha"] > h["fecha"].max() - dt.timedelta(days=dias)]

    def maximos_historicos(self):
        return {(t, f): float(m) for (t, f), m in
                self.historico.groupby(["tienda_id", "familia_id"])["ventas"].max().items()}

    def registrar_solicitud(self, sid, endpoint, cuerpo, codigo, tienda_id=None, fecha=None,
                            calidad=None, detalle=None, version=None, predicciones=None,
                            con=None):
        self.solicitudes.append({"id": sid, "endpoint": endpoint, "codigo": codigo,
                                 "calidad": calidad, "detalle": detalle,
                                 "predicciones": predicciones, "version": version})

    def registrar_observacion(self, sid, cuerpo, tienda_id, fecha, ventas, bitacora):
        cambios = 0
        for v in ventas:
            clave = (tienda_id, v["familia_id"], fecha)
            previo = self.observaciones.get(clave)
            if previo != v["ventas"]:
                if previo is not None:
                    bitacora.advertir("ventas", f"'{v['familia_id']}' se reemplazó")
                self.observaciones[clave] = v["ventas"]
                cambios += 1
        self.registrar_solicitud(sid, "/v1/observaciones", cuerpo, 200, tienda_id, fecha,
                                 bitacora.calidad())
        return cambios

    def crear_reentrenamiento(self, id_):
        self.reentrenos[id_] = {"id": id_, "estado": "en_curso", "promovido": None,
                                "version": None, "motivo": None}

    def borrar_reentrenamiento(self, id_):
        self.reentrenos.pop(id_, None)

    def obtener_reentrenamiento(self, id_):
        return self.reentrenos.get(id_)

    def finalizar_reentrenamiento(self, id_, estado, promovido, version, motivo, metricas=None):
        self.reentrenos[id_] = {"id": id_, "estado": estado, "promovido": promovido,
                                "version": version, "motivo": motivo}


class AirflowFalso:
    def __init__(self):
        self.disponible = True
        self.estados: dict[str, str] = {}

    def disparar(self, run_id):
        if not self.disponible:
            raise ErrorAirflow("Airflow no responde")
        self.estados[run_id] = "running"

    def estado(self, run_id):
        if not self.disponible:
            raise ErrorAirflow("Airflow no responde")
        return self.estados.get(run_id)


@pytest.fixture
def bundle():
    b = crear_bundle("baseline")
    b["factores"] = {(f, None): 1.1 for f, _ in FAMILIAS}
    return b


@pytest.fixture
def entorno(cat, bundle):
    repo, airflow = RepoMemoria(), AirflowFalso()
    estado_deps = {"mlflow": True}

    def chequeo_mlflow():
        if not estado_deps["mlflow"]:
            raise ConnectionError("MLflow caído")

    gestor = GestorModelo("pronostico_ventas", "campeon", intervalo=3600,
                          chequeos={"mlflow": chequeo_mlflow})
    gestor.fijar("3", bundle)
    app = crear_app(Dependencias(cat, repo, gestor, airflow, "pronostico_ventas", "campeon"))
    with TestClient(app) as cliente:
        gestor.sincronizar()
        yield {"cliente": cliente, "repo": repo, "airflow": airflow, "gestor": gestor,
               "deps": estado_deps}


@pytest.fixture
def rng():
    return np.random.default_rng(0)
