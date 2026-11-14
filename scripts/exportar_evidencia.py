"""Exporta evidencia de MLflow (runs, versiones, alias) y Airflow (ejecuciones del DAG).

    AIRFLOW_ADMIN_PASSWORD=... python3 scripts/exportar_evidencia.py [carpeta_salida]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

MLFLOW = "http://localhost:5000"
AIRFLOW = "http://localhost:8080"


def pedir(url, cuerpo=None, cabeceras=None, metodo=None):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(url, data=datos, method=metodo or ("POST" if datos else "GET"),
                                 headers={"Content-Type": "application/json", **(cabeceras or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    salida = Path(sys.argv[1] if len(sys.argv) > 1 else "evidencia")
    salida.mkdir(parents=True, exist_ok=True)
    lineas = ["# Evidencia de funcionamiento", ""]

    modelo = pedir(f"{MLFLOW}/api/2.0/mlflow/registered-models/get?name=pronostico_ventas")
    versiones = pedir(f"{MLFLOW}/api/2.0/mlflow/model-versions/search?filter="
                      "name%3D%27pronostico_ventas%27&max_results=100")["model_versions"]
    aliases = {a["version"]: a["alias"] for a in modelo["registered_model"].get("aliases", [])}
    lineas += ["## MLflow Model Registry: `pronostico_ventas`", "",
               "| Versión | Alias | Tipo | Promovido | Costo validación | Run |",
               "|---|---|---|---|---|---|"]
    for v in sorted(versiones, key=lambda v: int(v["version"])):
        tags = {t["key"]: t["value"] for t in v.get("tags", [])}
        lineas.append(f"| {v['version']} | {aliases.get(v['version'], '')} | "
                      f"{tags.get('tipo', '')} | {tags.get('promovido', '')} | "
                      f"{tags.get('costo_validacion', '')} | `{v.get('run_id', '')[:12]}` |")

    exp = pedir(f"{MLFLOW}/api/2.0/mlflow/experiments/get-by-name?"
                "experiment_name=novamercado_pronostico")["experiment"]
    runs = pedir(f"{MLFLOW}/api/2.0/mlflow/runs/search",
                 {"experiment_ids": [exp["experiment_id"]], "max_results": 200,
                  "order_by": ["attributes.start_time ASC"]})["runs"]
    lineas += ["", f"## MLflow: experimento `novamercado_pronostico` ({len(runs)} runs)", "",
               "| Run | Costo prueba | WAPE | MAE | Costo vigente | Decisión |",
               "|---|---|---|---|---|---|"]
    for r in runs:
        m = {x["key"]: x["value"] for x in r["data"].get("metrics", [])}
        t = {x["key"]: x["value"] for x in r["data"].get("tags", [])}
        costo = m.get("costo_total", m.get("candidato_costo_total"))
        wape = m.get("wape", m.get("candidato_wape"))
        mae = m.get("mae", m.get("candidato_mae"))
        vigente = m.get("vigente_costo_total")

        def fmt(x, patron):
            return "" if x is None else format(x, patron)

        lineas.append(f"| {r['info'].get('run_name', '')} | {fmt(costo, ',.0f')} | "
                      f"{fmt(wape, '.3f')} | {fmt(mae, '.1f')} | {fmt(vigente, ',.0f')} | "
                      f"{t.get('motivo', '')} |")

    clave = os.environ.get("AIRFLOW_ADMIN_PASSWORD")
    if clave:
        token = pedir(f"{AIRFLOW}/auth/token", {"username": "admin", "password": clave})
        cab = {"Authorization": f"Bearer {token['access_token']}"}
        runs_af = pedir(f"{AIRFLOW}/api/v2/dags/reentrenamiento_pronostico/dagRuns?limit=100",
                        cabeceras=cab)["dag_runs"]
        lineas += ["", f"## Airflow: ejecuciones de `reentrenamiento_pronostico` "
                       f"({len(runs_af)})", "",
                   "| dag_run_id | Estado | Inicio | Fin |", "|---|---|---|---|"]
        for d in runs_af:
            lineas.append(f"| {d['dag_run_id']} | {d['state']} | {d.get('start_date', '')} | "
                          f"{d.get('end_date', '')} |")
        (salida / "airflow_dag_runs.json").write_text(json.dumps(runs_af, indent=2))

    (salida / "mlflow_versiones.json").write_text(json.dumps(versiones, indent=2))
    (salida / "EVIDENCIA.md").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
