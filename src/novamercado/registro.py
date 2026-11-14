"""Model Registry de MLflow: registrar, promover (alias) y descargar modelos.

El modelo se guarda como pyfunc de MLflow cuyo artefacto principal es el bundle (joblib).
La API descarga el artefacto y lo carga directamente; guarda una copia local para seguir
operando si MLflow no está disponible.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path


log = logging.getLogger(__name__)

ARCHIVO_BUNDLE = "bundle.pkl"


def guardar_bundle(bundle: dict, ruta: Path) -> None:
    """Serializa con el pickle nativo (implementación en C).

    En la imagen de Airflow `dill` se registra en el pickler de Python que usa joblib y
    serializa partes de scikit-learn con tipos de dill; la API no tiene dill y no podría
    cargar el modelo. El pickler en C no se ve afectado. Se verifica igualmente.
    """
    import pickle
    datos = pickle.dumps(bundle, protocol=5)
    if b"dill" in datos:
        raise RuntimeError("el modelo serializado depende de dill; la API no podría cargarlo")
    Path(ruta).write_bytes(datos)


def cargar_bundle(ruta: Path) -> dict:
    import pickle
    return pickle.loads(Path(ruta).read_bytes())


def _mlflow():
    # Importación diferida: las pruebas de la API no necesitan MLflow.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    import mlflow
    return mlflow


def cliente(uri: str):
    mlflow = _mlflow()
    mlflow.set_tracking_uri(uri)
    return mlflow.MlflowClient(tracking_uri=uri)


def version_con_alias(uri: str, nombre: str, alias: str) -> str | None:
    """Versión que tiene el alias, o None si no existe. Lanza si MLflow no responde."""
    from mlflow.exceptions import MlflowException
    c = cliente(uri)
    try:
        return str(c.get_model_version_by_alias(nombre, alias).version)
    except MlflowException as e:
        codigo = getattr(e, "error_code", "")
        if codigo in ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE") or \
                "not found" in str(e).lower() or "does not exist" in str(e).lower():
            return None
        raise


def registrar_bundle(bundle: dict, nombre_modelo: str) -> str:
    """Registra el bundle como nueva versión (dentro del run activo). Devuelve la versión."""
    mlflow = _mlflow()
    import mlflow.pyfunc

    class ModeloPronostico(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.bundle = cargar_bundle(context.artifacts["bundle"])

        def predict(self, context, model_input, params=None):
            from novamercado.modelo import predecir_ventas
            return predecir_ventas(self.bundle, model_input)

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / ARCHIVO_BUNDLE
        guardar_bundle(bundle, ruta)
        info = mlflow.pyfunc.log_model(
            name="modelo",
            python_model=ModeloPronostico(),
            artifacts={"bundle": str(ruta)},
            registered_model_name=nombre_modelo,
            pip_requirements=["scikit-learn==1.9.1", "pandas==3.1.0", "numpy==2.5.4",
                              "joblib==1.6.0"],
        )
    return str(info.registered_model_version)


def asignar_alias(uri: str, nombre: str, alias: str, version: str) -> None:
    cliente(uri).set_registered_model_alias(nombre, alias, version)


def descargar_bundle(uri: str, nombre: str, version: str, cache_dir: Path) -> dict:
    """Descarga la versión a cache_dir/<version>/ (si no estaba) y carga el bundle."""
    destino = Path(cache_dir) / str(version)
    archivo = destino / ARCHIVO_BUNDLE
    if not archivo.exists():
        mlflow = _mlflow()
        mlflow.set_tracking_uri(uri)
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{nombre}/{version}", dst_path=tmp)
            encontrados = list(Path(local).rglob(ARCHIVO_BUNDLE))
            if not encontrados:
                raise FileNotFoundError(f"la versión {version} no contiene {ARCHIVO_BUNDLE}")
            destino.mkdir(parents=True, exist_ok=True)
            shutil.copy2(encontrados[0], archivo.with_suffix(".tmp"))
            os.replace(archivo.with_suffix(".tmp"), archivo)
    return cargar_bundle(archivo)


def guardar_vigente_local(cache_dir: Path, version: str) -> None:
    ruta = Path(cache_dir) / "vigente.json"
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": version}))
    os.replace(tmp, ruta)


def leer_vigente_local(cache_dir: Path) -> tuple[str, dict] | None:
    """Último modelo vigente guardado localmente (para arrancar sin MLflow)."""
    ruta = Path(cache_dir) / "vigente.json"
    try:
        version = json.loads(ruta.read_text())["version"]
        return version, cargar_bundle(Path(cache_dir) / version / ARCHIVO_BUNDLE)
    except Exception:  # noqa: BLE001 - sin caché válida se espera a MLflow
        return None
