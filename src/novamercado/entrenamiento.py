"""Reentrenamiento: entrena candidatos, los compara con el vigente y decide el reemplazo.

Ventana de evaluación (últimos `dias_validacion` días con ventas conocidas):

    |------------- entrenamiento -------------|--- calibración ---|----- prueba -----|
                                               (mitad de la ventana)  (mitad final)

* Los candidatos (baseline y gradient boosting) se entrenan solo con "entrenamiento".
* Los factores de la cantidad sugerida se calibran en "calibración".
* El costo de negocio se mide en "prueba", para el candidato y para el modelo vigente.

Criterio de reemplazo (DECISIONES.md, sección 4): se promueve el candidato si su costo en
prueba es al menos `margen_reemplazo` (1 %) menor que el del vigente, o si no hay vigente.
Se registra exactamente el modelo evaluado (no se reentrena con la ventana), de modo que la
comparación con el vigente es fuera de muestra para ambos. El alias se mueve solo al final:
si algo falla antes, el vigente no cambia.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import registro
from .catalogos import Catalogos
from .config import Config
from .features import construir_features
from .modelo import (PARAMETROS_HGB, calibrar_factores, crear_bundle, entrenar_hgb,
                     filas_entrenables, metricas, predecir_ventas, sugerir)

log = logging.getLogger(__name__)


def _miles(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def dividir(feats: pd.DataFrame, dias_validacion: int):
    fmax = feats["fecha"].max()
    c1 = fmax - pd.Timedelta(days=dias_validacion)
    c2 = fmax - pd.Timedelta(days=dias_validacion // 2)
    return (feats[feats["fecha"] <= c1],
            feats[(feats["fecha"] > c1) & (feats["fecha"] <= c2)],
            feats[feats["fecha"] > c2])


def evaluar_candidato(tipo: str, train, calib, test, cat: Catalogos):
    est = entrenar_hgb(train) if tipo == "hgb" else None
    bundle = crear_bundle(tipo, est)
    pred_cal = predecir_ventas(bundle, calib)
    bundle["factores"] = calibrar_factores(calib, pred_cal, cat)
    pred_test = predecir_ventas(bundle, test)
    m = metricas(test, pred_test, sugerir(bundle, test, pred_test), cat)
    return bundle, m, np.concatenate([pred_cal, pred_test])


def decidir(costo_candidato: float, costo_vigente: float | None, margen: float) -> tuple[bool, str]:
    if costo_vigente is None:
        return True, (f"No había modelo vigente; se promueve el candidato (costo en validación "
                      f"{_miles(costo_candidato)})")
    umbral = costo_vigente * (1 - margen)
    texto = (f"Costo en validación {_miles(costo_candidato)} vs {_miles(costo_vigente)} del "
             f"vigente")
    if costo_candidato < umbral:
        return True, texto
    return False, f"{texto}; no mejora al menos {margen:.0%}, se conserva el vigente"


def ejecutar_reentrenamiento(cfg: Config, repo, cat: Catalogos, run_id: str) -> dict:
    mlflow = registro._mlflow()
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experimento)

    ventas = repo.ventas_entrenamiento()
    n_obs = repo.conteo_observaciones()
    feats = filas_entrenables(construir_features(ventas, cat))
    train, calib, test = dividir(feats, cfg.dias_validacion)
    log.info("datos: %d filas (%d observaciones registradas); hasta %s", len(feats), n_obs,
             feats["fecha"].max().date())

    version_vigente = registro.version_con_alias(cfg.mlflow_uri, cfg.nombre_modelo, cfg.alias)
    bundle_vigente = None
    if version_vigente:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_vigente = registro.descargar_bundle(cfg.mlflow_uri, cfg.nombre_modelo,
                                                       version_vigente, Path(tmp))

    with mlflow.start_run(run_name=run_id) as run:
        mlflow.set_tags({"reentrenamiento_id": run_id, "version_vigente_previa":
                         version_vigente or "ninguna"})
        mlflow.log_params({
            "fecha_max_datos": str(feats["fecha"].max().date()),
            "fecha_min_datos": str(feats["fecha"].min().date()),
            "n_filas": len(feats), "n_observaciones_registradas": n_obs,
            "dias_validacion": cfg.dias_validacion, "margen_reemplazo": cfg.margen_reemplazo,
            "inicio_prueba": str(test["fecha"].min().date()),
        })

        candidatos = {}
        for tipo in ("baseline", "hgb"):
            with mlflow.start_run(run_name=f"{run_id}-{tipo}", nested=True):
                mlflow.set_tag("tipo_modelo", tipo)
                if tipo == "hgb":
                    mlflow.log_params({f"hgb_{k}": v for k, v in PARAMETROS_HGB.items()})
                bundle, m, pred_ventana = evaluar_candidato(tipo, train, calib, test, cat)
                mlflow.log_metrics(m)
                candidatos[tipo] = (bundle, m, pred_ventana)
                log.info("%s: costo prueba %.0f, WAPE %.3f", tipo, m["costo_total"], m["wape"])

        tipo_ganador = min(candidatos, key=lambda t: candidatos[t][1]["costo_total"])
        m_cand = candidatos[tipo_ganador][1]
        mlflow.log_metrics({f"candidato_{k}": v for k, v in m_cand.items()})

        costo_vigente = None
        if bundle_vigente is not None:
            try:
                pv = predecir_ventas(bundle_vigente, test)
                m_vig = metricas(test, pv, sugerir(bundle_vigente, test, pv), cat)
                costo_vigente = m_vig["costo_total"]
                mlflow.log_metrics({f"vigente_{k}": v for k, v in m_vig.items()})
            except Exception as e:  # noqa: BLE001 - un vigente que no predice se reemplaza
                log.warning("el modelo vigente no pudo evaluarse: %s", e)
                costo_vigente = float("inf")

        # Se registra exactamente el modelo evaluado (entrenado antes de la ventana). Así el
        # vigente nunca vio los días de prueba de un reentrenamiento posterior y la
        # comparación es siempre fuera de muestra para ambos.
        bundle_final = candidatos[tipo_ganador][0]
        bundle_final["metadatos"] = {
            "entrenado_hasta": str(train["fecha"].max().date()),
            "factores_calibrados_hasta": str(calib["fecha"].max().date()),
            "run_id": run.info.run_id, "reentrenamiento_id": run_id,
        }
        version_nueva = registro.registrar_bundle(bundle_final, cfg.nombre_modelo)

        promovido, motivo = decidir(m_cand["costo_total"], costo_vigente, cfg.margen_reemplazo)
        if costo_vigente == float("inf"):
            motivo = "El modelo vigente no pudo evaluarse; se promueve el candidato"
        motivo = f"{motivo} (candidato {tipo_ganador}, versión {version_nueva})"
        mlflow.set_tags({"tipo_ganador": tipo_ganador, "version_registrada": version_nueva,
                         "promovido": str(promovido), "motivo": motivo})
        cliente = registro.cliente(cfg.mlflow_uri)
        cliente.set_model_version_tag(cfg.nombre_modelo, version_nueva, "promovido", str(promovido))
        cliente.set_model_version_tag(cfg.nombre_modelo, version_nueva, "costo_validacion",
                                      f"{m_cand['costo_total']:.1f}")
        cliente.set_model_version_tag(cfg.nombre_modelo, version_nueva, "tipo", tipo_ganador)
        if promovido:
            registro.asignar_alias(cfg.mlflow_uri, cfg.nombre_modelo, cfg.alias, version_nueva)
        version_final = version_nueva if promovido else version_vigente

    log.info("resultado: promovido=%s versión vigente=%s; %s", promovido, version_final, motivo)
    return {"promovido": promovido, "version": version_final, "motivo": motivo,
            "metricas": {"candidato": m_cand,
                         "costo_vigente": costo_vigente if costo_vigente is not None
                         and np.isfinite(costo_vigente) else None,
                         "version_registrada": version_nueva}}
