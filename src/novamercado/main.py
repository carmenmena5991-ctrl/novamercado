"""Punto de entrada de producción: `uvicorn novamercado.main:app --port 8000`."""
from __future__ import annotations

import logging

from .airflow_cliente import ClienteAirflow
from .api import Dependencias, crear_app
from .catalogos import cargar_catalogos
from .config import cargar_config
from .db import RepositorioPostgres
from .registro import (descargar_bundle, guardar_vigente_local, leer_vigente_local,
                       version_con_alias)
from .servicio import GestorModelo, chequeo_airflow, chequeo_http, crear_cache_dir, esperar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("novamercado")

cfg = cargar_config()
cache = crear_cache_dir(cfg.cache_dir)
repo = RepositorioPostgres(cfg.db_url)


def _al_iniciar() -> None:
    def _esquema() -> bool:
        try:
            repo.crear_esquema()
            return True
        except Exception as e:  # noqa: BLE001
            log.info("esperando a la base de datos: %s", e)
            return False
    esperar(_esquema, 120, paso=3)


gestor = GestorModelo(
    nombre=cfg.nombre_modelo, alias=cfg.alias, intervalo=cfg.intervalo_sondeo,
    version_remota=lambda: version_con_alias(cfg.mlflow_uri, cfg.nombre_modelo, cfg.alias),
    descargar=lambda v: descargar_bundle(cfg.mlflow_uri, cfg.nombre_modelo, v, cache),
    cache_local=lambda: leer_vigente_local(cache),
    guardar_local=lambda v: guardar_vigente_local(cache, v),
    chequeos={
        "base_de_datos": repo.ping,
        "mlflow": chequeo_http(f"{cfg.mlflow_uri}/health"),
        "airflow": chequeo_airflow(cfg.airflow_url),
    },
)

app = crear_app(Dependencias(
    catalogos=cargar_catalogos(cfg.datos_dir), repo=repo, gestor=gestor,
    airflow=ClienteAirflow(cfg.airflow_url, cfg.airflow_usuario, cfg.airflow_password,
                           cfg.dag_id),
    nombre_modelo=cfg.nombre_modelo, alias=cfg.alias, al_iniciar=_al_iniciar,
))
