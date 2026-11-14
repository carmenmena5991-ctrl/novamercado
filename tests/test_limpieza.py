"""Políticas ante datos defectuosos (DECISIONES.md, sección 1)."""
from __future__ import annotations

import datetime as dt

import pytest

from novamercado.limpieza import (Rechazo, limpiar_observacion, limpiar_prediccion,
                                  parsear_fecha, parsear_numero)

from .conftest import observacion_valida, solicitud_valida


def acciones(bit):
    return [(o["campo"], o["accion"]) for o in bit.observaciones]


# ------------------------------------------------------------------- caso sin problemas
def test_solicitud_valida_queda_ok(cat):
    limpio, bit = limpiar_prediccion(solicitud_valida(), cat)
    assert bit.estado == "ok" and bit.observaciones == []
    assert limpio["tienda_id"] == 5 and limpio["fecha"] == dt.date(2025, 11, 17)
    assert len(limpio["familias"]) == 12


# --------------------------------------------------------------------------- E01 tienda
@pytest.mark.parametrize("valor", ["T05", "5", " 5 ", "t-5", "Tienda 5", 5.0, "#5"])
def test_e01_tienda_reparable(cat, valor):
    limpio, bit = limpiar_prediccion(solicitud_valida(tienda_id=valor), cat)
    assert limpio["tienda_id"] == 5
    assert bit.estado == "corregido"
    assert acciones(bit) == [("tienda_id", "reparar")]


@pytest.mark.parametrize("valor", [99, "T99", "5-6", 5.5, True, None, "abc", [5]])
def test_e01_tienda_irreparable_se_rechaza(cat, valor):
    with pytest.raises(Rechazo):
        limpiar_prediccion(solicitud_valida(tienda_id=valor), cat)


# -------------------------------------------------------------------------- E02 familia
@pytest.mark.parametrize("valor,esperado", [
    ("LÁCTEOS", "lacteos"), ("lacteos ", "lacteos"), ("Frutas y Verduras", "frutas_verduras"),
    ("frutas-verduras", "frutas_verduras"), ("Snacks y dulces", "snacks"), ("BEBÉ", "bebe"),
    ("Cuidado Personal", "cuidado_personal"),
])
def test_e02_familia_otra_forma_se_repara(cat, valor, esperado):
    s = solicitud_valida()
    idx = next(i for i, f in enumerate(s["familias"]) if f["familia_id"] == esperado)
    s["familias"][idx]["familia_id"] = valor
    limpio, bit = limpiar_prediccion(s, cat)
    assert esperado in [f["familia_id"] for f in limpio["familias"]]
    assert (f"familias[{idx}].familia_id", "reparar") in acciones(bit)
    assert bit.estado == "corregido"


def test_familia_con_error_de_tipeo_se_advierte(cat):
    s = solicitud_valida()
    s["familias"][0]["familia_id"] = "abarotes"
    limpio, bit = limpiar_prediccion(s, cat)
    assert limpio["familias"][0]["familia_id"] == "abarrotes"
    assert ("familias[0].familia_id", "advertir") in acciones(bit)
    assert bit.estado == "advertencia"


def test_familia_desconocida_se_ignora_y_se_completa(cat):
    s = solicitud_valida()
    s["familias"][0]["familia_id"] = "electronica"
    limpio, bit = limpiar_prediccion(s, cat)
    assert bit.estado == "advertencia"
    # abarrotes se pronostica igual, como familia faltante con promoción 0
    assert {"familia_id": "abarrotes", "en_promocion": 0} in limpio["familias"]


# ----------------------------------------------------------------------- E03 promoción
@pytest.mark.parametrize("valor,esperado,accion", [
    ("3", 3, "reparar"), (2.0, 2, "reparar"), (" 4 ", 4, "reparar"),
    (2.5, 2, "advertir"), (-1, 0, "advertir"), ("tres", 0, "advertir"), (None, 0, "advertir"),
])
def test_e03_promocion(cat, valor, esperado, accion):
    s = solicitud_valida()
    s["familias"][1]["en_promocion"] = valor
    limpio, bit = limpiar_prediccion(s, cat)
    assert limpio["familias"][1]["en_promocion"] == esperado
    assert ("familias[1].en_promocion", accion) in acciones(bit)


def test_promocion_ausente_se_supone_cero(cat):
    s = solicitud_valida()
    del s["familias"][2]["en_promocion"]
    limpio, bit = limpiar_prediccion(s, cat)
    assert limpio["familias"][2]["en_promocion"] == 0
    assert bit.estado == "advertencia"


# ---------------------------------------------------------------------------- E04 fecha
@pytest.mark.parametrize("valor,ambigua", [
    ("2025-11-17T00:00:00", False), ("17/11/2025", False), ("2025/11/17", False),
    ("20251117", False), ("11/17/2025", False), ("17-11-2025", False),
    ("17 de noviembre de 2025", False), ("2025-11-17T08:30:00Z", False),
])
def test_e04_fecha_otro_formato(valor, ambigua):
    f = parsear_fecha(valor)
    assert f.valor == dt.date(2025, 11, 17)
    assert f.ambigua is ambigua and f.nota


def test_fecha_ambigua_se_advierte_dia_mes(cat):
    limpio, bit = limpiar_prediccion(solicitud_valida(fecha="05/11/2025"), cat)
    assert limpio["fecha"] == dt.date(2025, 11, 5)
    assert ("fecha", "advertir") in acciones(bit)


@pytest.mark.parametrize("valor", ["2025-02-30", "mañana", "", None, "31/31/2025", 3.5])
def test_fecha_invalida_se_rechaza(cat, valor):
    with pytest.raises(Rechazo):
        limpiar_prediccion(solicitud_valida(fecha=valor), cat)


def test_fecha_fuera_del_calendario_se_rechaza_sin_usar_reloj(cat):
    with pytest.raises(Rechazo, match="calendario"):
        limpiar_prediccion(solicitud_valida(fecha="2027-01-05"), cat)
    limpio, _ = limpiar_prediccion(solicitud_valida(fecha="2026-12-31"), cat)
    assert limpio["fecha"] == dt.date(2026, 12, 31)


def test_fecha_antes_de_apertura_de_tienda_se_rechaza(cat):
    with pytest.raises(Rechazo, match="abrió"):
        limpiar_prediccion(solicitud_valida(tienda_id=10, fecha="2025-01-10"), cat)


# ------------------------------------------------------------------ E05 faltan familias
def test_e05_faltan_familias_se_completan_con_advertencia(cat):
    s = solicitud_valida()
    s["familias"] = [f for f in s["familias"] if f["familia_id"] not in ("bebe", "mascotas")]
    limpio, bit = limpiar_prediccion(s, cat)
    assert len(limpio["familias"]) == 12
    assert ("familias", "advertir") in acciones(bit)
    assert "bebe" in bit.observaciones[0]["detalle"]


# ---------------------------------------------------------------- E06 familia repetida
def test_e06_repetida_con_valores_distintos_usa_el_ultimo(cat):
    s = solicitud_valida()
    s["familias"].append({"familia_id": "abarrotes", "en_promocion": 3})
    limpio, bit = limpiar_prediccion(s, cat)
    abarrotes = [f for f in limpio["familias"] if f["familia_id"] == "abarrotes"]
    assert abarrotes == [{"familia_id": "abarrotes", "en_promocion": 3}]
    assert ("familias", "advertir") in acciones(bit)


def test_repetida_identica_se_repara(cat):
    s = solicitud_valida()
    s["familias"].append({"familia_id": "abarrotes", "en_promocion": 0})
    limpio, bit = limpiar_prediccion(s, cat)
    assert len(limpio["familias"]) == 12
    assert bit.estado == "corregido"


# ---------------------------------------------------------------- E07 campo renombrado
@pytest.mark.parametrize("clave", ["tienda", "id_tienda", "Tienda_ID", "store_id", "tienda_di"])
def test_e07_campo_renombrado_se_repara(cat, clave):
    s = solicitud_valida()
    s[clave] = s.pop("tienda_id")
    limpio, bit = limpiar_prediccion(s, cat)
    assert limpio["tienda_id"] == 5
    assert ("tienda_id", "reparar") in acciones(bit)


def test_campos_en_elementos_renombrados(cat):
    s = solicitud_valida()
    s["familias"][0] = {"familia": "abarrotes", "promocion": 1}
    limpio, bit = limpiar_prediccion(s, cat)
    assert limpio["familias"][0] == {"familia_id": "abarrotes", "en_promocion": 1}
    assert bit.estado == "corregido"


def test_campo_desconocido_se_ignora_con_advertencia(cat):
    limpio, bit = limpiar_prediccion(solicitud_valida(canal="web"), cat)
    assert ("canal", "advertir") in acciones(bit)


def test_campo_obligatorio_ausente_se_rechaza(cat):
    s = solicitud_valida()
    del s["fecha"]
    with pytest.raises(Rechazo, match="fecha"):
        limpiar_prediccion(s, cat)


def test_sinonimos_contradictorios_se_rechazan(cat):
    s = solicitud_valida()
    del s["tienda_id"]
    s["tienda"], s["store_id"] = 5, 6
    with pytest.raises(Rechazo):
        limpiar_prediccion(s, cat)


@pytest.mark.parametrize("cuerpo", [[], "texto", 5, None])
def test_cuerpo_no_objeto_se_rechaza(cat, cuerpo):
    with pytest.raises(Rechazo):
        limpiar_prediccion(cuerpo, cat)


def test_familias_vacias_se_rechaza(cat):
    with pytest.raises(Rechazo):
        limpiar_prediccion(solicitud_valida(familias=[]), cat)


def test_familias_como_diccionario_se_repara(cat):
    limpio, bit = limpiar_prediccion(solicitud_valida(familias={"abarrotes": 2, "bebidas": 0}),
                                     cat)
    assert {"familia_id": "abarrotes", "en_promocion": 2} in limpio["familias"]


# ---------------------------------------------------------------------- observaciones
def test_observacion_valida_ok(cat):
    limpio, bit = limpiar_observacion(observacion_valida(), cat)
    assert bit.estado == "ok" and len(limpio["ventas"]) == 12


def test_e08_ventas_negativas_no_se_registran(cat):
    o = observacion_valida()
    o["ventas"][5]["ventas"] = -4
    limpio, bit = limpiar_observacion(o, cat)
    assert len(limpio["ventas"]) == 11
    assert ("ventas[5].ventas", "advertir") in acciones(bit)


@pytest.mark.parametrize("valor,esperado,accion", [
    ("1,234", 1234, "reparar"), ("812,0", 812, "reparar"), ("50,0", 50, "reparar"),
    ("1.234.567", 1234567, "reparar"), ("1,234.0", 1234, "reparar"), ("1.234,0", 1234, "reparar"),
    ("1.234", 1234, "advertir"), (" 994 ", 994, "reparar"), (994.0, 994, "reparar"),
    ("12,5", 12, "advertir"),
])
def test_e10_numero_con_formato(cat, valor, esperado, accion):
    o = observacion_valida()
    o["ventas"][0]["ventas"] = valor
    limpio, bit = limpiar_observacion(o, cat)
    assert limpio["ventas"][0]["ventas"] == esperado
    assert ("ventas[0].ventas", accion) in acciones(bit)


def test_observacion_valor_atipico_no_se_registra(cat):
    o = observacion_valida()
    o["ventas"][0]["ventas"] = 999_999
    limpio, bit = limpiar_observacion(o, cat, maximos={(5, "abarrotes"): 1000.0})
    assert "abarrotes" not in [v["familia_id"] for v in limpio["ventas"]]
    assert bit.estado == "advertencia"


def test_observacion_duplicada_contradictoria_se_omite(cat):
    o = observacion_valida()
    o["ventas"].append({"familia_id": "abarrotes", "ventas": 5})
    limpio, bit = limpiar_observacion(o, cat)
    assert "abarrotes" not in [v["familia_id"] for v in limpio["ventas"]]


def test_observacion_sin_ventas_validas_se_rechaza(cat):
    o = observacion_valida(ventas=[{"familia_id": "abarrotes", "ventas": -1}])
    with pytest.raises(Rechazo):
        limpiar_observacion(o, cat)


@pytest.mark.parametrize("valor", ["", "abc", None, True, [], {}, "1,2,3", float("nan")])
def test_numero_invalido(valor):
    assert parsear_numero(valor) is None
