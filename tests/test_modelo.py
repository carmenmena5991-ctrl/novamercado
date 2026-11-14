"""Variables sin fuga de información, costos y decisión de reemplazo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from novamercado.entrenamiento import decidir
from novamercado.features import construir_features
from novamercado.modelo import (calibrar_factores, costo, crear_bundle, entrenar_hgb,
                                filas_entrenables, predecir_ventas)


def _serie(cat, dias=120, rng=None):
    filas = []
    for d in pd.date_range("2025-06-01", periods=dias):
        for f in ("abarrotes", "panaderia"):
            v = 100 + 30 * (d.weekday() >= 5) + (rng.normal(0, 5) if rng is not None else 0)
            filas.append({"tienda_id": 5, "familia_id": f, "fecha": d, "ventas": max(v, 0),
                          "en_promocion": 0})
    return pd.DataFrame(filas)


def test_features_no_usan_la_venta_del_mismo_dia(cat):
    df = _serie(cat)
    base = construir_features(df, cat)
    df2 = df.copy()
    ultimo = df2["fecha"] == df2["fecha"].max()
    df2.loc[ultimo, "ventas"] = 10_000  # cambiar la venta del día no cambia sus variables
    alterado = construir_features(df2, cat)
    cols = ["lag_1", "media_7", "media_28", "mismo_dia_4", "base"]
    a = base[base["fecha"] == df["fecha"].max()][cols].reset_index(drop=True)
    b = alterado[alterado["fecha"] == df["fecha"].max()][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_feriado_local_solo_aplica_a_su_ciudad(cat):
    df = pd.DataFrame({"tienda_id": [8, 5], "familia_id": ["abarrotes"] * 2,
                       "fecha": ["2026-08-04"] * 2, "ventas": [1.0, 1.0], "en_promocion": [0, 0]})
    f = construir_features(df, cat).set_index("tienda_id")
    assert f.loc[8, "es_feriado"] == 1  # Cobán
    assert f.loc[5, "es_feriado"] == 0  # Quetzaltenango


def test_costo_asimetrico():
    c = costo(np.array([10, 10]), np.array([8, 13]), np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert list(c) == [2.0, 6.0]


def test_factor_mayor_cuando_faltar_cuesta_mas(cat, rng):
    feats = filas_entrenables(construir_features(_serie(cat, rng=rng), cat))
    pred = feats["media_28"].to_numpy()
    fac = calibrar_factores(feats, pred, cat)
    # abarrotes: faltante 1.0 / sobrante 0.2 -> sobre-stock; panadería 1.0 / 2.5 -> bajo
    assert fac[("abarrotes", "hipermercado")] > fac[("panaderia", "hipermercado")]


def test_hgb_mejora_baseline_en_serie_con_patron(cat, rng):
    feats = filas_entrenables(construir_features(_serie(cat, dias=200, rng=rng), cat))
    corte = feats["fecha"].max() - pd.Timedelta(days=28)
    train, test = feats[feats["fecha"] <= corte], feats[feats["fecha"] > corte]
    hgb = crear_bundle("hgb", entrenar_hgb(train))
    err_hgb = np.abs(predecir_ventas(hgb, test) - test["ventas"]).mean()
    err_base = np.abs(predecir_ventas(crear_bundle("baseline"), test) - test["ventas"]).mean()
    assert err_hgb <= err_base * 1.1


def test_bundle_se_serializa_sin_dill(cat, rng, tmp_path):
    from novamercado.registro import cargar_bundle, guardar_bundle
    feats = filas_entrenables(construir_features(_serie(cat, rng=rng), cat))
    b = crear_bundle("hgb", entrenar_hgb(feats), {("abarrotes", None): 1.2})
    guardar_bundle(b, tmp_path / "b.pkl")
    assert b"dill" not in (tmp_path / "b.pkl").read_bytes()
    otro = cargar_bundle(tmp_path / "b.pkl")
    np.testing.assert_allclose(predecir_ventas(otro, feats), predecir_ventas(b, feats))


def test_criterio_de_reemplazo():
    assert decidir(100, None, 0.01)[0] is True
    assert decidir(98, 100, 0.01)[0] is True
    assert decidir(99.5, 100, 0.01)[0] is False
    promovido, motivo = decidir(18210, 19940, 0.01)
    assert promovido and motivo == "Costo en validación 18 210 vs 19 940 del vigente"
