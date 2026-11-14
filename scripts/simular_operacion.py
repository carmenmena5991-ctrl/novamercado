"""Simula la operación diaria posterior al histórico (solo biblioteca estándar).

    python3 scripts/simular_operacion.py data/publicados [dias] [inicio] [cambio_desde_dia]

Ventas "reales" simuladas: las del mismo día de la semana un año antes (364 días),
multiplicadas por 1.3 desde `cambio_desde_dia` (cambio de demanda). Cada día:
1) observaciones del día anterior; 2) pronóstico de cada tienda; 3) costo de la sugerencia.
Cada 7 días dispara un reentrenamiento y espera su resultado.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

API = "http://localhost:8000"


def llamar(metodo, ruta, cuerpo=None):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(f"{API}{ruta}", data=datos, method=metodo,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    datos = Path(sys.argv[1])
    dias = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    inicio = dt.date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else dt.date(2026, 3, 1)
    cambio = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    hist = defaultdict(dict)  # fecha -> (tienda, familia) -> (ventas, promo)
    with open(datos / "historico_ventas.csv") as f:
        for r in csv.DictReader(f):
            hist[r["fecha"]][(int(r["tienda_id"]), r["familia_id"])] = (
                int(r["ventas"]), int(r["en_promocion"]))
    costos = {}
    with open(datos / "costos.csv") as f:
        for r in csv.DictReader(f):
            costos[r["familia_id"]] = (float(r["costo_faltante"]), float(r["costo_sobrante"]))

    def real(fecha: dt.date, n_dia: int):
        factor = 1.3 if n_dia >= cambio else 1.0
        salida = {}
        for t in range(1, 11):
            # Mismo día de la semana un año antes; la tienda 10 (abrió en 2025) usa la
            # misma semana más reciente disponible en el histórico.
            for atras in (364, 35, 42, 49, 56, 63):
                base = hist.get((fecha - dt.timedelta(days=atras)).isoformat(), {})
                filas = {k: v for k, v in base.items() if k[0] == t}
                if filas:
                    break
            salida.update({k: (round(v * factor), p) for k, (v, p) in filas.items()})
        return salida

    costo_total, versiones = 0.0, defaultdict(int)
    for n in range(dias):
        fecha = inicio + dt.timedelta(days=n)
        if n > 0:  # observaciones del día anterior
            previo = fecha - dt.timedelta(days=1)
            por_tienda = defaultdict(list)
            for (t, fam), (v, _) in real(previo, n - 1).items():
                por_tienda[t].append({"familia_id": fam, "ventas": v})
            for t, ventas in por_tienda.items():
                codigo, _ = llamar("POST", "/v1/observaciones",
                                   {"tienda_id": t, "fecha": previo.isoformat(), "ventas": ventas})
                assert codigo == 200, codigo
        reales = real(fecha, n)
        costo_dia = 0.0
        for t in range(1, 11):
            familias = [{"familia_id": fam, "en_promocion": p}
                        for (tt, fam), (_, p) in reales.items() if tt == t]
            codigo, resp = llamar("POST", "/v1/prediccion", {"tienda_id": t,
                                  "fecha": fecha.isoformat(), "familias": familias})
            assert codigo == 200, resp
            versiones[resp["version_modelo"]] += 1
            for p in resp["predicciones"]:
                v = reales[(t, p["familia_id"])][0]
                cf, cs = costos[p["familia_id"]]
                s = p["cantidad_sugerida"]
                costo_dia += cf * max(v - s, 0) + cs * max(s - v, 0)
        costo_total += costo_dia
        print(f"{fecha} costo {costo_dia:9.0f}  versión {resp['version_modelo']}"
              f"{'  (demanda x1.3)' if n >= cambio else ''}", flush=True)

        if n % 7 == 6:
            _, r = llamar("POST", "/admin/reentrenar")
            t0 = time.time()
            while True:
                time.sleep(5)
                _, estado = llamar("GET", f"/admin/reentrenar/{r['id']}")
                if estado["estado"] != "en_curso":
                    break
            print(f"  reentrenamiento {r['id']} ({time.time() - t0:.0f} s): {estado}", flush=True)
            if estado.get("promovido"):
                t1 = time.time()
                while llamar("GET", "/modelo")[1]["version"] != estado["version"]:
                    time.sleep(2)
                print(f"  la API usa la versión {estado['version']} tras {time.time() - t1:.0f} s")
    print(f"costo total {costo_total:.0f}; predicciones por versión {dict(versiones)}")


if __name__ == "__main__":
    main()
