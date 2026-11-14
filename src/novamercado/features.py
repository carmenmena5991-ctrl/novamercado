"""Variables del modelo. La MISMA función se usa al entrenar y al predecir.

Información que usa el modelo al pronosticar la fecha F (DECISIONES.md, sección 2):
* ventas de días anteriores a F de la misma tienda y familia (nunca de F ni posteriores);
* promociones de F, que llegan en la solicitud;
* calendario de F: día de la semana, mes, feriados nacionales y de la ciudad de la tienda,
  fechas comerciales y de pago (feriados.csv);
* atributos fijos de la tienda (tipo) y de la familia.

Las ventanas cuentan los últimos N días *observados*, no N días de calendario: si faltan
observaciones, el modelo usa las más recientes disponibles y lo informa (advertencia).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogos import Catalogos

TIPOS_TIENDA = ["express", "supermercado", "hipermercado"]
TIPOS_FERIADO = ["feriado", "comercial", "pago", "medio_dia"]

COLUMNAS_MODELO = [
    "familia_cod", "tienda_cod", "tipo_cod",
    "dia_semana", "dia_mes", "mes", "dia_anio", "fin_de_mes", "quincena",
    "es_feriado", "es_comercial", "es_pago", "es_medio_dia", "vispera_feriado",
    "post_feriado", "dias_a_navidad",
    "en_promocion", "promo_rel",
    "lag_1_rel", "media_7_rel", "mismo_dia_4_rel", "std_28_rel", "tendencia",
    "log_base", "dias_desde_ultimo",
]
CATEGORICAS = ["familia_cod", "tienda_cod", "tipo_cod"]


def _calendario(fechas: pd.Series, ciudades: pd.Series, cat: Catalogos) -> pd.DataFrame:
    fer = cat.feriados
    # (fecha, ámbito) -> tipos
    tipos_por: dict[tuple, set[str]] = {}
    for f, tipo, amb in zip(fer["fecha"], fer["tipo"], fer["ambito"]):
        tipos_por.setdefault((pd.Timestamp(f), amb), set()).add(tipo)

    def tipos(f: pd.Timestamp, ciudad: str) -> set[str]:
        return tipos_por.get((f, "nacional"), set()) | tipos_por.get((f, ciudad), set())

    claves = pd.DataFrame({"fecha": fechas.values, "ciudad": ciudades.values})
    unicas = claves.drop_duplicates()
    filas = []
    un_dia = pd.Timedelta(days=1)
    for f, c in zip(unicas["fecha"], unicas["ciudad"]):
        t = tipos(f, c)
        navidad = pd.Timestamp(year=f.year, month=12, day=25)
        filas.append({
            "fecha": f, "ciudad": c,
            "es_feriado": int("feriado" in t), "es_comercial": int("comercial" in t),
            "es_pago": int("pago" in t), "es_medio_dia": int("medio_dia" in t),
            "vispera_feriado": int("feriado" in tipos(f + un_dia, c)),
            "post_feriado": int("feriado" in tipos(f - un_dia, c)),
            "dias_a_navidad": int(np.clip((navidad - f).days, -10, 40)),
        })
    tabla = pd.DataFrame(filas)
    out = claves.merge(tabla, on=["fecha", "ciudad"], how="left")
    out.index = fechas.index
    return out.drop(columns=["fecha", "ciudad"])


def construir_features(ventas: pd.DataFrame, cat: Catalogos) -> pd.DataFrame:
    """`ventas`: columnas tienda_id, familia_id, fecha, ventas (NaN en filas a pronosticar),
    en_promocion (puede ser NaN). Devuelve las mismas filas con las variables del modelo y
    `base` (nivel de referencia); el objetivo es ventas / base."""
    df = ventas.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values(["tienda_id", "familia_id", "fecha"]).reset_index(drop=True)
    g = df.groupby(["tienda_id", "familia_id"], sort=False)["ventas"]
    previo = g.shift(1)
    df["lag_1"] = previo
    claves = [df["tienda_id"], df["familia_id"]]
    df["media_7"] = previo.groupby(claves).transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["media_28"] = previo.groupby(claves).transform(lambda s: s.rolling(28, min_periods=1).mean())
    df["std_28"] = previo.groupby(claves).transform(lambda s: s.rolling(28, min_periods=2).std())
    df["dia_semana"] = df["fecha"].dt.weekday
    previo_dow = df.groupby(["tienda_id", "familia_id", "dia_semana"], sort=False)["ventas"].shift(1)
    df["mismo_dia_4"] = previo_dow.groupby([df["tienda_id"], df["familia_id"], df["dia_semana"]]) \
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    fecha_prev = df.groupby(["tienda_id", "familia_id"], sort=False)["fecha"].shift(1)
    df["dias_desde_ultimo"] = (df["fecha"] - fecha_prev).dt.days

    df["base"] = df["media_28"].clip(lower=1.0)
    df["lag_1_rel"] = df["lag_1"] / df["base"]
    df["media_7_rel"] = df["media_7"] / df["base"]
    df["mismo_dia_4_rel"] = df["mismo_dia_4"] / df["base"]
    df["std_28_rel"] = df["std_28"] / df["base"]
    df["tendencia"] = df["media_7"] / df["base"]
    df["log_base"] = np.log1p(df["media_28"])

    # Promoción del día relativa a las de los 28 días previos (ventana igual al predecir).
    promo_prev = df.groupby(["tienda_id", "familia_id"], sort=False)["en_promocion"].shift(1)
    promo_media = promo_prev.groupby(claves).transform(
        lambda s: s.rolling(28, min_periods=1).mean())
    df["promo_rel"] = df["en_promocion"] / (promo_media.fillna(0) + 1.0)

    df["dia_mes"] = df["fecha"].dt.day
    df["mes"] = df["fecha"].dt.month
    df["dia_anio"] = df["fecha"].dt.dayofyear
    df["fin_de_mes"] = (df["fecha"].dt.is_month_end | (df["dia_mes"] <= 2)).astype(int)
    df["quincena"] = df["dia_mes"].between(14, 16).astype(int)

    tiendas = cat.tiendas
    df["tipo"] = df["tienda_id"].map(tiendas["tipo"])
    df["ciudad"] = df["tienda_id"].map(tiendas["ciudad"])
    df["familia_cod"] = df["familia_id"].map({f: i for i, f in enumerate(cat.ids_familia)})
    df["tienda_cod"] = df["tienda_id"].astype(int)
    df["tipo_cod"] = df["tipo"].map({t: i for i, t in enumerate(TIPOS_TIENDA)}).fillna(-1)
    cal = _calendario(df["fecha"], df["ciudad"], cat)
    df = pd.concat([df, cal], axis=1)
    return df
