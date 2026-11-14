"""Estado del servicio: modelo vigente con recarga en caliente y salud de dependencias.

Un hilo en segundo plano, cada `intervalo` segundos:
* comprueba MLflow, Airflow y la base de datos;
* si MLflow responde, lee la versión con el alias vigente y, si cambió, la descarga y la
  activa sin reiniciar (cambio atómico de referencia);
* si MLflow no responde, sigue sirviendo el modelo cargado (o el de la caché local).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

log = logging.getLogger(__name__)


@dataclass
class ModeloCargado:
    version: str
    bundle: dict


class GestorModelo:
    def __init__(self, nombre: str, alias: str, intervalo: int = 15,
                 version_remota: Callable[[], str | None] | None = None,
                 descargar: Callable[[str], dict] | None = None,
                 cache_local: Callable[[], tuple[str, dict] | None] | None = None,
                 guardar_local: Callable[[str], None] | None = None,
                 chequeos: dict[str, Callable[[], None]] | None = None):
        self.nombre, self.alias, self.intervalo = nombre, alias, intervalo
        self._version_remota = version_remota
        self._descargar = descargar
        self._cache_local = cache_local
        self._guardar_local = guardar_local
        self._chequeos = chequeos or {}
        self._actual: ModeloCargado | None = None
        self._estado_deps: dict[str, str | None] = {k: "sin verificar" for k in self._chequeos}
        self._lock = threading.Lock()
        self._detener = threading.Event()
        self._hilo: threading.Thread | None = None

    # ------------------------------------------------------------------------ lectura
    @property
    def actual(self) -> ModeloCargado | None:
        return self._actual

    def fijar(self, version: str, bundle: dict) -> None:
        self._actual = ModeloCargado(version, bundle)

    def problemas(self) -> dict[str, str]:
        return {k: v for k, v in self._estado_deps.items() if v}

    # ----------------------------------------------------------------------- ciclo
    def iniciar(self) -> None:
        if self._actual is None and self._cache_local:
            local = self._cache_local()
            if local:
                self.fijar(*local)
                log.info("modelo versión %s cargado desde la caché local", local[0])
        self._hilo = threading.Thread(target=self._ciclo, name="gestor-modelo", daemon=True)
        self._hilo.start()

    def detener(self) -> None:
        self._detener.set()

    def _ciclo(self) -> None:
        while not self._detener.is_set():
            self.sincronizar()
            self._detener.wait(self.intervalo)

    def sincronizar(self) -> None:
        with self._lock:
            for nombre, chequeo in self._chequeos.items():
                try:
                    chequeo()
                    self._estado_deps[nombre] = None
                except Exception as e:  # noqa: BLE001
                    self._estado_deps[nombre] = f"{type(e).__name__}: {str(e)[:150]}"
            if self._version_remota is None or self._estado_deps.get("mlflow"):
                return
            try:
                version = self._version_remota()
                if version and (self._actual is None or self._actual.version != version):
                    bundle = self._descargar(version)
                    self.fijar(version, bundle)
                    if self._guardar_local:
                        self._guardar_local(version)
                    log.info("modelo vigente actualizado a la versión %s", version)
            except Exception as e:  # noqa: BLE001
                self._estado_deps["mlflow"] = f"no se pudo sincronizar el modelo: {str(e)[:150]}"
                log.warning("no se pudo sincronizar el modelo: %s", e)


def chequeo_http(url: str, timeout: float = 3.0) -> Callable[[], None]:
    def _chequear() -> None:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    return _chequear


def chequeo_airflow(url_base: str, timeout: float = 3.0) -> Callable[[], None]:
    def _chequear() -> None:
        r = requests.get(f"{url_base}/api/v2/monitor/health", timeout=timeout)
        r.raise_for_status()
        sched = r.json().get("scheduler", {}).get("status")
        if sched not in (None, "healthy"):
            raise RuntimeError(f"scheduler de Airflow: {sched}")
    return _chequear


def crear_cache_dir(ruta: Path) -> Path:
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def esperar(condicion: Callable[[], bool], segundos: float, paso: float = 2.0) -> bool:
    fin = time.monotonic() + segundos
    while time.monotonic() < fin:
        if condicion():
            return True
        time.sleep(paso)
    return condicion()
