"""Contrato HTTP del Anexo, sin MLflow ni Airflow reales."""
from __future__ import annotations

import pytest

from .conftest import observacion_valida, solicitud_valida


def test_prediccion_valida_cumple_contrato(entorno):
    r = entorno["cliente"].post("/v1/prediccion", json=solicitud_valida())
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) >= {"tienda_id", "fecha", "version_modelo", "predicciones", "calidad"}
    assert cuerpo["tienda_id"] == 5 and cuerpo["fecha"] == "2025-11-17"
    assert cuerpo["version_modelo"] == "3" and isinstance(cuerpo["version_modelo"], str)
    assert len(cuerpo["predicciones"]) == 12
    for p in cuerpo["predicciones"]:
        assert set(p) == {"familia_id", "venta_esperada", "cantidad_sugerida"}
        assert isinstance(p["cantidad_sugerida"], int) and p["cantidad_sugerida"] >= 0
        assert isinstance(p["venta_esperada"], float)
    assert cuerpo["calidad"] == {"estado": "ok", "observaciones": []}
    registro = entorno["repo"].solicitudes[-1]
    assert registro["codigo"] == 200 and registro["version"] == "3"
    assert len(registro["predicciones"]) == 12


def test_prediccion_corregida_informa_campo(entorno):
    s = solicitud_valida()
    s["familias"][2]["familia_id"] = "LÁCTEOS"
    cuerpo = entorno["cliente"].post("/v1/prediccion", json=s).json()
    assert cuerpo["calidad"]["estado"] == "corregido"
    assert cuerpo["calidad"]["observaciones"][0]["campo"] == "familias[2].familia_id"
    assert cuerpo["calidad"]["observaciones"][0]["accion"] == "reparar"


def test_prediccion_con_advertencia(entorno):
    s = solicitud_valida()
    s["familias"] = s["familias"][:10]
    cuerpo = entorno["cliente"].post("/v1/prediccion", json=s).json()
    assert cuerpo["calidad"]["estado"] == "advertencia"
    assert len(cuerpo["predicciones"]) == 12


def test_rechazo_responde_422_con_detail_y_se_registra(entorno):
    r = entorno["cliente"].post("/v1/prediccion", json=solicitud_valida(tienda_id=77))
    assert r.status_code == 422
    assert "77" in r.json()["detail"]
    assert entorno["repo"].solicitudes[-1]["codigo"] == 422


@pytest.mark.parametrize("ruta", ["/v1/prediccion", "/v1/observaciones"])
@pytest.mark.parametrize("cuerpo", [b"{no es json", b"null", b""])
def test_cuerpo_ilegible_responde_422_con_detail_texto_y_se_registra(entorno, ruta, cuerpo):
    r = entorno["cliente"].post(ruta, content=cuerpo,
                                headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]
    registro = entorno["repo"].solicitudes[-1]
    assert registro["endpoint"] == ruta and registro["codigo"] == 422


def test_misma_fecha_puede_pedirse_dos_veces(entorno):
    c = entorno["cliente"]
    assert c.post("/v1/prediccion", json=solicitud_valida()).status_code == 200
    assert c.post("/v1/prediccion", json=solicitud_valida()).status_code == 200


def test_sin_historia_previa_se_rechaza(entorno):
    r = entorno["cliente"].post("/v1/prediccion", json=solicitud_valida(fecha="2025-01-10"))
    assert r.status_code == 422


def test_hueco_de_datos_se_advierte(entorno):
    cuerpo = entorno["cliente"].post("/v1/prediccion",
                                     json=solicitud_valida(fecha="2026-01-20")).json()
    assert cuerpo["calidad"]["estado"] == "advertencia"
    assert any(o["campo"] == "fecha" for o in cuerpo["calidad"]["observaciones"])


def test_cantidad_sugerida_aplica_factor(entorno):
    cuerpo = entorno["cliente"].post("/v1/prediccion", json=solicitud_valida()).json()
    for p in cuerpo["predicciones"]:
        assert p["cantidad_sugerida"] == round(p["venta_esperada"] * 1.1) or \
            abs(p["cantidad_sugerida"] - p["venta_esperada"] * 1.1) <= 1


# ---------------------------------------------------------------------- observaciones
def test_observaciones_idempotentes(entorno):
    c = entorno["cliente"]
    r1 = c.post("/v1/observaciones", json=observacion_valida())
    assert r1.status_code == 200
    assert r1.json() == {"registros_guardados": 12, "calidad": {"estado": "ok", "observaciones": []}}
    r2 = c.post("/v1/observaciones", json=observacion_valida())
    assert r2.json()["registros_guardados"] == 0
    assert len(entorno["repo"].observaciones) == 12


def test_observacion_modificada_cuenta_y_advierte(entorno):
    c = entorno["cliente"]
    c.post("/v1/observaciones", json=observacion_valida())
    o = observacion_valida()
    o["ventas"][0]["ventas"] = 1
    r = c.post("/v1/observaciones", json=o).json()
    assert r["registros_guardados"] == 1
    assert r["calidad"]["estado"] == "advertencia"


def test_observacion_con_formato_numerico(entorno):
    o = observacion_valida()
    o["ventas"][1]["ventas"] = "1,292"
    r = entorno["cliente"].post("/v1/observaciones", json=o).json()
    assert r["calidad"]["estado"] == "corregido"
    assert entorno["repo"].observaciones[(5, "bebidas", __import__("datetime").date(2025, 11, 17))] == 1292


# ------------------------------------------------------------------- administración
def test_reentrenar_ciclo_de_estados(entorno):
    c, repo, airflow = entorno["cliente"], entorno["repo"], entorno["airflow"]
    r = c.post("/admin/reentrenar")
    assert r.status_code == 202
    run_id = r.json()["id"]
    assert isinstance(run_id, str)
    assert c.get(f"/admin/reentrenar/{run_id}").json() == {"estado": "en_curso"}
    repo.finalizar_reentrenamiento(run_id, "exitoso", True, "4", "Costo 1 vs 2")
    airflow.estados[run_id] = "success"
    assert c.get(f"/admin/reentrenar/{run_id}").json() == {
        "estado": "exitoso", "promovido": True, "version": "4", "motivo": "Costo 1 vs 2"}


def test_reentrenamiento_fallido_en_airflow(entorno):
    c, airflow = entorno["cliente"], entorno["airflow"]
    run_id = c.post("/admin/reentrenar").json()["id"]
    airflow.estados[run_id] = "failed"
    cuerpo = c.get(f"/admin/reentrenar/{run_id}").json()
    assert cuerpo["estado"] == "fallido" and cuerpo["promovido"] is False
    assert cuerpo["version"] == "3"


def test_reentrenar_sin_airflow_responde_503(entorno):
    entorno["airflow"].disponible = False
    assert entorno["cliente"].post("/admin/reentrenar").status_code == 503


def test_id_desconocido_404(entorno):
    assert entorno["cliente"].get("/admin/reentrenar/no-existe").status_code == 404


def test_modelo_vigente(entorno):
    assert entorno["cliente"].get("/modelo").json() == {
        "nombre": "pronostico_ventas", "alias": "campeon", "version": "3"}


# ------------------------------------------------------------------------------ salud
def test_health_ok_y_degradado(entorno):
    c, gestor = entorno["cliente"], entorno["gestor"]
    assert c.get("/health").json()["estado"] == "ok"
    entorno["deps"]["mlflow"] = False
    gestor.sincronizar()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "degradado" and "mlflow" in r.json()["motivo"]
    # La API sigue pronosticando con el modelo cargado.
    assert c.post("/v1/prediccion", json=solicitud_valida()).status_code == 200
    entorno["deps"]["mlflow"] = True
    gestor.sincronizar()
    assert c.get("/health").json()["estado"] == "ok"


def test_health_sin_modelo_503(entorno):
    entorno["gestor"]._actual = None
    assert entorno["cliente"].get("/health").status_code == 503
    assert entorno["cliente"].post("/v1/prediccion", json=solicitud_valida()).status_code == 503


def test_recarga_en_caliente_de_version(entorno, bundle):
    gestor = entorno["gestor"]
    gestor._version_remota = lambda: "4"
    gestor._descargar = lambda v: bundle
    gestor.sincronizar()
    cuerpo = entorno["cliente"].post("/v1/prediccion", json=solicitud_valida()).json()
    assert cuerpo["version_modelo"] == "4"


def test_promocion_inverosimil_se_acota_y_se_advierte(entorno):
    s = solicitud_valida()
    s["familias"][0]["en_promocion"] = 1e12
    cuerpo = entorno["cliente"].post("/v1/prediccion", json=s).json()
    assert cuerpo["calidad"]["estado"] == "advertencia"
    registro = entorno["repo"].solicitudes[-1]
    assert registro["predicciones"][0]["en_promocion"] == 1000


def test_cuerpo_con_nan_se_guarda_como_json_valido_para_postgres():
    import json
    from novamercado.db import _jsonable, texto_seguro
    nul = chr(0)
    limpio = _jsonable({"a": float("nan"), "b": [float("inf")], "c" + nul: "x" + nul})
    texto = json.dumps(limpio, allow_nan=False)  # lanzaría si quedara NaN/Infinity
    assert nul not in texto
    assert limpio["a"] == "nan" and limpio["b"] == ["inf"]
    assert nul not in texto_seguro("campo" + nul)
