"""Arranque desde cero (servicio `bootstrap` de Compose), idempotente:

1. crea el esquema de la base del producto;
2. carga historico_ventas.csv si no se había cargado;
3. si no existe un modelo con el alias vigente en MLflow, entrena y registra el primero.

Al volver a levantar el producto, los pasos 2 y 3 detectan lo existente y no se repiten.
"""
from __future__ import annotations

import logging
import sys

import requests

from .catalogos import cargar_catalogos
from .config import cargar_config
from .db import RepositorioPostgres
from .entrenamiento import ejecutar_reentrenamiento
from .registro import version_con_alias
from .servicio import esperar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("bootstrap")


def main() -> int:
    cfg = cargar_config()
    cat = cargar_catalogos(cfg.datos_dir)
    repo = RepositorioPostgres(cfg.db_url)

    def _db() -> bool:
        try:
            repo.crear_esquema()
            return True
        except Exception as e:  # noqa: BLE001
            log.info("esperando a la base de datos: %s", e)
            return False

    if not esperar(_db, 180, paso=3):
        log.error("la base de datos no respondió")
        return 1
    n = repo.cargar_historico(cfg.datos_dir / "historico_ventas.csv")
    log.info("histórico: %s", f"{n} filas cargadas" if n else "ya estaba cargado")

    def _mlflow() -> bool:
        try:
            return requests.get(f"{cfg.mlflow_uri}/health", timeout=3).ok
        except requests.RequestException:
            return False

    if not esperar(_mlflow, 300, paso=5):
        log.error("MLflow no respondió")
        return 1
    version = version_con_alias(cfg.mlflow_uri, cfg.nombre_modelo, cfg.alias)
    if version:
        log.info("ya existe el modelo %s@%s (versión %s); no se entrena", cfg.nombre_modelo,
                 cfg.alias, version)
        return 0
    log.info("no hay modelo vigente: entrenando el primero")
    res = ejecutar_reentrenamiento(cfg, repo, cat, "inicial")
    repo.finalizar_reentrenamiento("inicial", "exitoso", res["promovido"], res["version"],
                                   res["motivo"], res["metricas"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
