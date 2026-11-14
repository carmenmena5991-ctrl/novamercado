"""Cliente mínimo de la API REST v2 de Airflow 3 (disparar y consultar el DAG)."""
from __future__ import annotations

import threading
import time

import requests


class ErrorAirflow(Exception):
    pass


class ClienteAirflow:
    def __init__(self, url: str, usuario: str, password: str, dag_id: str, timeout: float = 10):
        self.url, self.usuario, self.password = url.rstrip("/"), usuario, password
        self.dag_id, self.timeout = dag_id, timeout
        self._token: str | None = None
        self._vence = 0.0
        self._lock = threading.Lock()

    def _cabeceras(self, renovar: bool = False) -> dict:
        with self._lock:
            if renovar or not self._token or time.time() > self._vence:
                r = requests.post(f"{self.url}/auth/token",
                                  json={"username": self.usuario, "password": self.password},
                                  timeout=self.timeout)
                if r.status_code >= 400:
                    raise ErrorAirflow(f"autenticación con Airflow falló ({r.status_code})")
                self._token = r.json()["access_token"]
                self._vence = time.time() + 600
            return {"Authorization": f"Bearer {self._token}"}

    def _pedir(self, metodo: str, ruta: str, **kw) -> requests.Response:
        try:
            r = requests.request(metodo, f"{self.url}{ruta}", headers=self._cabeceras(),
                                 timeout=self.timeout, **kw)
            if r.status_code == 401:
                r = requests.request(metodo, f"{self.url}{ruta}",
                                     headers=self._cabeceras(renovar=True),
                                     timeout=self.timeout, **kw)
            return r
        except requests.RequestException as e:
            raise ErrorAirflow(f"Airflow no responde: {e}") from e

    def disparar(self, run_id: str) -> None:
        r = self._pedir("POST", f"/api/v2/dags/{self.dag_id}/dagRuns",
                        json={"dag_run_id": run_id, "logical_date": None,
                              "conf": {"reentrenamiento_id": run_id}})
        if r.status_code >= 400:
            raise ErrorAirflow(f"Airflow rechazó la ejecución ({r.status_code}): {r.text[:200]}")

    def estado(self, run_id: str) -> str | None:
        """queued | running | success | failed | None si no existe."""
        r = self._pedir("GET", f"/api/v2/dags/{self.dag_id}/dagRuns/{run_id}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise ErrorAirflow(f"Airflow respondió {r.status_code}")
        return r.json().get("state")
