"""Prueba de punta a punta contra el producto levantado (solo biblioteca estándar).

    python3 scripts/prueba_integral.py [carpeta_ejemplos] [carpeta_salida]

Envía las solicitudes válidas y defectuosas de data/publicados/ejemplos/, verifica
idempotencia de observaciones y guarda cada respuesta en evidencia/respuestas/.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:8000"


def llamar(metodo: str, ruta: str, cuerpo=None) -> tuple[int, dict]:
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(f"{API}{ruta}", data=datos, method=metodo,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main() -> None:
    ejemplos = Path(sys.argv[1] if len(sys.argv) > 1 else "data/publicados/ejemplos")
    salida = Path(sys.argv[2] if len(sys.argv) > 2 else "evidencia/respuestas")
    salida.mkdir(parents=True, exist_ok=True)

    archivos = sorted(ejemplos.glob("*.json")) + sorted((ejemplos / "errores").glob("*.json"))
    for archivo in archivos:
        cuerpo = json.loads(archivo.read_text())
        ruta = "/v1/observaciones" if "ventas" in cuerpo and "familias" not in cuerpo \
            else "/v1/prediccion"
        codigo, resp = llamar("POST", ruta, cuerpo)
        estado = resp.get("calidad", {}).get("estado", resp.get("detail", ""))
        print(f"{archivo.name:32s} {ruta:18s} {codigo}  {str(estado)[:70]}")
        (salida / archivo.name).write_text(json.dumps(
            {"solicitud": cuerpo, "endpoint": ruta, "codigo_http": codigo, "respuesta": resp},
            ensure_ascii=False, indent=2))

    # Problemas fuera del catálogo
    extra = {
        "X01_tienda_inexistente": ("/v1/prediccion", {"tienda_id": 42, "fecha": "2026-03-02",
                                    "familias": [{"familia_id": "abarrotes", "en_promocion": 0}]}),
        "X02_fecha_fuera_calendario": ("/v1/prediccion", {"tienda_id": 5, "fecha": "2031-01-01",
                                        "familias": [{"familia_id": "abarrotes", "en_promocion": 0}]}),
        "X03_familias_como_objeto": ("/v1/prediccion", {"tienda_id": 5, "fecha": "2026-03-02",
                                      "familias": {"abarrotes": 1, "bebidas": "2"}}),
        "X04_fecha_ambigua": ("/v1/prediccion", {"tienda_id": 5, "fecha": "02/03/2026",
                               "familias": [{"familia_id": "abarrotes", "en_promocion": 0}]}),
        "X05_campos_en_elemento": ("/v1/prediccion", {"store": "T-5", "date": "2026-03-02",
                                    "familias": [{"familia": "Panadería", "promo": "1"}]}),
        "X06_observacion_texto": ("/v1/observaciones", {"tienda_id": 5, "fecha": "2026-03-01",
                                   "ventas": [{"familia_id": "abarrotes", "ventas": "abc"},
                                              {"familia_id": "bebidas", "ventas": "1.234"}]}),
        "X07_cuerpo_lista": ("/v1/prediccion", [1, 2, 3]),
    }
    for nombre, (ruta, cuerpo) in extra.items():
        codigo, resp = llamar("POST", ruta, cuerpo)
        estado = resp.get("calidad", {}).get("estado", resp.get("detail", ""))
        print(f"{nombre:32s} {ruta:18s} {codigo}  {str(estado)[:70]}")
        (salida / f"{nombre}.json").write_text(json.dumps(
            {"solicitud": cuerpo, "endpoint": ruta, "codigo_http": codigo, "respuesta": resp},
            ensure_ascii=False, indent=2))

    # Idempotencia: la misma observación dos veces
    obs = json.loads((ejemplos / "observacion_valida.json").read_text())
    obs["fecha"] = "2026-03-01"
    primera = llamar("POST", "/v1/observaciones", obs)[1]["registros_guardados"]
    segunda = llamar("POST", "/v1/observaciones", obs)[1]["registros_guardados"]
    print(f"idempotencia: primer envío {primera} registros, reenvío {segunda}")
    assert segunda == 0

    for ruta in ("/modelo", "/health"):
        codigo, resp = llamar("GET", ruta)
        print(ruta, codigo, resp)
        (salida / f"GET{ruta.replace('/', '_')}.json").write_text(
            json.dumps(resp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
