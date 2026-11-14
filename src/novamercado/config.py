"""Configuración leída de variables de entorno (ver .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    datos_dir: Path
    db_url: str
    mlflow_uri: str
    nombre_modelo: str
    alias: str
    experimento: str
    airflow_url: str
    airflow_usuario: str
    airflow_password: str
    dag_id: str
    cache_dir: Path
    intervalo_sondeo: int
    # Criterio de reemplazo: el candidato debe costar al menos este % menos que el vigente.
    margen_reemplazo: float
    dias_validacion: int


def cargar_config() -> Config:
    e = os.environ.get
    return Config(
        datos_dir=Path(e("NOVA_DATOS_DIR", "/datos")),
        db_url=e("NOVA_DB_URL", "postgresql://nova:nova@db:5432/novamercado"),
        mlflow_uri=e("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        nombre_modelo=e("NOVA_NOMBRE_MODELO", "pronostico_ventas"),
        alias=e("NOVA_ALIAS_MODELO", "campeon"),
        experimento=e("NOVA_EXPERIMENTO", "novamercado_pronostico"),
        airflow_url=e("NOVA_AIRFLOW_URL", "http://airflow-apiserver:8080"),
        airflow_usuario=e("NOVA_AIRFLOW_USUARIO", "admin"),
        airflow_password=e("NOVA_AIRFLOW_PASSWORD", "admin"),
        dag_id=e("NOVA_DAG_ID", "reentrenamiento_pronostico"),
        cache_dir=Path(e("NOVA_CACHE_DIR", "/cache_modelo")),
        intervalo_sondeo=int(e("NOVA_INTERVALO_SONDEO", "15")),
        margen_reemplazo=float(e("NOVA_MARGEN_REEMPLAZO", "0.01")),
        dias_validacion=int(e("NOVA_DIAS_VALIDACION", "56")),
    )
