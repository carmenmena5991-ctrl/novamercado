"""DAG de reentrenamiento del pronóstico de ventas.

Se dispara desde la API (POST /admin/reentrenar) con conf={"reentrenamiento_id": ...}; el
run_id del DAG es ese mismo id. También puede lanzarse a mano desde la interfaz de Airflow.

    verificar_datos -> entrenar_evaluar_registrar -> registrar_resultado

El alias del modelo vigente solo se mueve dentro de `entrenar_evaluar_registrar`, después de
evaluar; si cualquier tarea falla, el callback anota la ejecución como fallida y el modelo
vigente no cambia.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import pendulum
from airflow.sdk import dag, get_current_context, task

log = logging.getLogger(__name__)


def _id_ejecucion(context) -> str:
    conf = (context["dag_run"].conf or {})
    return conf.get("reentrenamiento_id") or context["dag_run"].run_id


def _al_fallar(context) -> None:
    from novamercado.config import cargar_config
    from novamercado.db import RepositorioPostgres
    from novamercado.registro import version_con_alias
    cfg = cargar_config()
    try:
        version = version_con_alias(cfg.mlflow_uri, cfg.nombre_modelo, cfg.alias)
    except Exception:  # noqa: BLE001 - MLflow caído: la versión no se conoce
        version = None
    tarea = context.get("task_instance")
    motivo = (f"La tarea {getattr(tarea, 'task_id', '?')} falló "
              f"({str(context.get('exception'))[:200]}); el modelo vigente no cambió")
    try:
        RepositorioPostgres(cfg.db_url).finalizar_reentrenamiento(
            _id_ejecucion(context), "fallido", False, version, motivo)
    except Exception as e:  # noqa: BLE001
        log.error("no se pudo registrar la falla: %s", e)


@dag(
    dag_id="reentrenamiento_pronostico",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=10),
    default_args={"retries": 0, "on_failure_callback": _al_fallar,
                  "execution_timeout": timedelta(minutes=9)},
    tags=["novamercado", "mlops"],
    doc_md=__doc__,
)
def reentrenamiento_pronostico():

    @task
    def verificar_datos() -> dict:
        from novamercado.config import cargar_config
        from novamercado.db import RepositorioPostgres
        cfg = cargar_config()
        repo = RepositorioPostgres(cfg.db_url)
        repo.crear_esquema()
        n_obs = repo.conteo_observaciones()
        if repo.metadato("historico_sha256") is None:
            raise RuntimeError("el histórico no está cargado en la base del producto")
        log.info("observaciones registradas: %d", n_obs)
        return {"observaciones": n_obs}

    @task
    def entrenar_evaluar_registrar(_datos: dict) -> dict:
        from novamercado.catalogos import cargar_catalogos
        from novamercado.config import cargar_config
        from novamercado.db import RepositorioPostgres
        from novamercado.entrenamiento import ejecutar_reentrenamiento
        cfg = cargar_config()
        ctx = get_current_context()
        res = ejecutar_reentrenamiento(cfg, RepositorioPostgres(cfg.db_url),
                                       cargar_catalogos(cfg.datos_dir), _id_ejecucion(ctx))
        return res

    @task
    def registrar_resultado(res: dict) -> None:
        from novamercado.config import cargar_config
        from novamercado.db import RepositorioPostgres
        cfg = cargar_config()
        ctx = get_current_context()
        RepositorioPostgres(cfg.db_url).finalizar_reentrenamiento(
            _id_ejecucion(ctx), "exitoso", res["promovido"], res["version"], res["motivo"],
            res["metricas"])

    registrar_resultado(entrenar_evaluar_registrar(verificar_datos()))


reentrenamiento_pronostico()
