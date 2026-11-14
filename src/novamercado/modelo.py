"""Modelos de pronóstico, cantidad sugerida y métricas de negocio.

Un "bundle" es un diccionario serializable con todo lo necesario para predecir:
    {"tipo": "baseline" | "hgb", "estimador": ..., "columnas": [...],
     "factores": {(familia_id, tipo_tienda): k}, "metadatos": {...}}

Cantidad sugerida = redondeo(venta_esperada * k[familia, tipo]). k se elige minimizando el
costo de costos.csv sobre predicciones fuera de muestra (DECISIONES.md, sección 3): es la
solución del problema del vendedor de periódicos (newsvendor) con la distribución empírica
del error, sin suponer una forma paramétrica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .catalogos import Catalogos
from .features import CATEGORICAS, COLUMNAS_MODELO

GRILLA_FACTORES = np.round(np.arange(0.70, 2.001, 0.02), 2)

PARAMETROS_HGB = dict(
    loss="squared_error", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
    min_samples_leaf=40, l2_regularization=0.1, random_state=42,
)


# ------------------------------------------------------------------------- predicción
def predecir_ventas(bundle: dict, feats: pd.DataFrame) -> np.ndarray:
    if bundle["tipo"] == "baseline":
        pred = feats["mismo_dia_4"].fillna(feats["media_28"])
    else:
        ratio = bundle["estimador"].predict(feats[bundle["columnas"]])
        pred = pd.Series(ratio, index=feats.index) * feats["base"]
    return np.clip(pred.fillna(0.0).to_numpy(dtype=float), 0.0, None)


def factores_de(bundle: dict, feats: pd.DataFrame) -> np.ndarray:
    fac = bundle.get("factores", {})
    return np.array([fac.get((f, t), fac.get((f, None), 1.0))
                     for f, t in zip(feats["familia_id"], feats["tipo"])], dtype=float)


def cantidades_sugeridas(pred: np.ndarray, factores: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.round(pred * factores)).astype(int)


def sugerir(bundle: dict, feats: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    return cantidades_sugeridas(pred, factores_de(bundle, feats))


# --------------------------------------------------------------------------- costos
def vector_costos(familias: pd.Series, cat: Catalogos) -> tuple[np.ndarray, np.ndarray]:
    cf = familias.map(cat.costos["costo_faltante"]).to_numpy(dtype=float)
    cs = familias.map(cat.costos["costo_sobrante"]).to_numpy(dtype=float)
    return cf, cs


def costo(real: np.ndarray, sugerido: np.ndarray, cf: np.ndarray, cs: np.ndarray) -> np.ndarray:
    real = np.asarray(real, dtype=float)
    sugerido = np.asarray(sugerido, dtype=float)
    return cf * np.maximum(real - sugerido, 0) + cs * np.maximum(sugerido - real, 0)


def calibrar_factores(feats: pd.DataFrame, pred: np.ndarray, cat: Catalogos) -> dict:
    """Para cada (familia, tipo de tienda) elige el multiplicador de menor costo."""
    real = feats["ventas"].to_numpy(dtype=float)
    cf, cs = vector_costos(feats["familia_id"], cat)
    factores: dict = {}
    grupos = pd.DataFrame({"f": feats["familia_id"].to_numpy(), "t": feats["tipo"].to_numpy()})
    for (fam, tipo), idx in grupos.groupby(["f", "t"]).indices.items():
        costos_k = [costo(real[idx], cantidades_sugeridas(pred[idx], k), cf[idx], cs[idx]).sum()
                    for k in GRILLA_FACTORES]
        factores[(fam, tipo)] = float(GRILLA_FACTORES[int(np.argmin(costos_k))])
    # Respaldo por familia (tipo de tienda no visto).
    for fam, idx in grupos.groupby("f").indices.items():
        costos_k = [costo(real[idx], cantidades_sugeridas(pred[idx], k), cf[idx], cs[idx]).sum()
                    for k in GRILLA_FACTORES]
        factores[(fam, None)] = float(GRILLA_FACTORES[int(np.argmin(costos_k))])
    return factores


def metricas(feats: pd.DataFrame, pred: np.ndarray, sugerido: np.ndarray,
             cat: Catalogos) -> dict[str, float]:
    real = feats["ventas"].to_numpy(dtype=float)
    cf, cs = vector_costos(feats["familia_id"], cat)
    c = costo(real, sugerido, cf, cs)
    err = pred - real
    dias = feats["fecha"].nunique()
    return {
        "costo_total": float(c.sum()),
        "costo_por_dia": float(c.sum() / max(dias, 1)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(np.abs(err).sum() / max(real.sum(), 1.0)),
        "sesgo": float(err.sum() / max(real.sum(), 1.0)),
        "tasa_quiebre": float((real > sugerido).mean()),
        "n_filas": float(len(real)),
    }


# --------------------------------------------------------------------------- entrenamiento
def filas_entrenables(feats: pd.DataFrame) -> pd.DataFrame:
    return feats[feats["ventas"].notna() & feats["media_28"].notna()]


def entrenar_hgb(feats: pd.DataFrame, parametros: dict | None = None) -> HistGradientBoostingRegressor:
    datos = filas_entrenables(feats)
    X = datos[COLUMNAS_MODELO]
    y = datos["ventas"] / datos["base"]
    modelo = HistGradientBoostingRegressor(
        categorical_features=[c in CATEGORICAS for c in COLUMNAS_MODELO],
        **(parametros or PARAMETROS_HGB),
    )
    # Peso = nivel de la serie: el error en ratio se pondera como error en unidades.
    modelo.fit(X, y, sample_weight=datos["base"].to_numpy())
    return modelo


def crear_bundle(tipo: str, estimador=None, factores: dict | None = None,
                 metadatos: dict | None = None) -> dict:
    return {"tipo": tipo, "estimador": estimador, "columnas": list(COLUMNAS_MODELO),
            "factores": factores or {}, "metadatos": metadatos or {}}
