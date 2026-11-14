"""Pronóstico de una solicitud canónica con el bundle vigente."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalogos import Catalogos
from .features import construir_features
from .limpieza import DIAS_SIN_DATOS_TOLERADOS, Bitacora, Rechazo
from .modelo import predecir_ventas, sugerir


def pronosticar(bundle: dict, solicitud: dict, historial: pd.DataFrame, cat: Catalogos,
                bit: Bitacora) -> list[dict]:
    tienda, fecha = solicitud["tienda_id"], solicitud["fecha"]
    if historial.empty:
        raise Rechazo(f"no hay ventas registradas de la tienda {tienda} antes de "
                      f"{fecha.isoformat()}; no es posible pronosticar sin inventar datos")
    ultima = pd.to_datetime(historial["fecha"]).max().date()
    hueco = (fecha - ultima).days
    if hueco > DIAS_SIN_DATOS_TOLERADOS:
        bit.advertir("fecha", f"el último dato de ventas de la tienda es del {ultima.isoformat()} "
                              f"({hueco} días antes); el pronóstico supone que el nivel de "
                              f"ventas no cambió desde entonces")

    objetivo = pd.DataFrame([{"tienda_id": tienda, "familia_id": f["familia_id"], "fecha": fecha,
                              "ventas": np.nan, "en_promocion": f["en_promocion"]}
                             for f in solicitud["familias"]])
    hist = historial[["tienda_id", "familia_id", "fecha", "ventas", "en_promocion"]].copy()
    hist["ventas"] = hist["ventas"].astype(float)
    hist["en_promocion"] = pd.to_numeric(hist["en_promocion"], errors="coerce")
    feats = construir_features(pd.concat([hist, objetivo], ignore_index=True), cat)
    feats = feats[(feats["fecha"] == pd.Timestamp(fecha)) & feats["ventas"].isna()]
    feats = feats.set_index("familia_id", drop=False)

    sin_historia = feats.index[feats["media_28"].isna()].tolist()
    for fam in sin_historia:
        bit.advertir("familias", f"'{fam}' no tiene ventas previas en la tienda; no se pronosticó")
    feats = feats.drop(index=sin_historia)
    if feats.empty:
        raise Rechazo("ninguna familia solicitada tiene ventas previas en la tienda")

    pred = predecir_ventas(bundle, feats)
    sug = sugerir(bundle, feats, pred)
    por_familia = {f: (float(p), int(s), int(pr)) for f, p, s, pr in
                   zip(feats.index, pred, sug, feats["en_promocion"])}
    salida = []
    for f in solicitud["familias"]:
        if f["familia_id"] in por_familia:
            p, s, pr = por_familia[f["familia_id"]]
            salida.append({"familia_id": f["familia_id"], "venta_esperada": round(p, 1),
                           "cantidad_sugerida": s, "en_promocion": pr})
    return salida
