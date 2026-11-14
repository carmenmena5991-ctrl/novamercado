"""Persistencia en PostgreSQL (base propia del producto, separada de la de Airflow)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

log = logging.getLogger(__name__)

ESQUEMA = """
CREATE TABLE IF NOT EXISTS metadatos (
    clave TEXT PRIMARY KEY,
    valor TEXT NOT NULL,
    actualizado_en TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS historico (
    tienda_id INT NOT NULL,
    familia_id TEXT NOT NULL,
    fecha DATE NOT NULL,
    ventas DOUBLE PRECISION NOT NULL,
    en_promocion INT,
    PRIMARY KEY (tienda_id, familia_id, fecha)
);
CREATE TABLE IF NOT EXISTS solicitudes (
    id UUID PRIMARY KEY,
    endpoint TEXT NOT NULL,
    recibida_en TIMESTAMPTZ NOT NULL DEFAULT now(),
    cuerpo JSONB,
    tienda_id INT,
    fecha DATE,
    codigo_http INT NOT NULL,
    estado_calidad TEXT,
    detalle TEXT,
    version_modelo TEXT
);
CREATE TABLE IF NOT EXISTS correcciones (
    id BIGSERIAL PRIMARY KEY,
    solicitud_id UUID NOT NULL REFERENCES solicitudes(id),
    campo TEXT NOT NULL,
    accion TEXT NOT NULL,
    detalle TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predicciones (
    id BIGSERIAL PRIMARY KEY,
    solicitud_id UUID NOT NULL REFERENCES solicitudes(id),
    tienda_id INT NOT NULL,
    fecha DATE NOT NULL,
    familia_id TEXT NOT NULL,
    en_promocion INT NOT NULL,
    venta_esperada DOUBLE PRECISION NOT NULL,
    cantidad_sugerida INT NOT NULL,
    version_modelo TEXT NOT NULL,
    creado_en TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS predicciones_clave ON predicciones (tienda_id, familia_id, fecha, id DESC);
CREATE TABLE IF NOT EXISTS observaciones (
    tienda_id INT NOT NULL,
    familia_id TEXT NOT NULL,
    fecha DATE NOT NULL,
    ventas DOUBLE PRECISION NOT NULL,
    solicitud_id UUID REFERENCES solicitudes(id),
    creado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
    actualizado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tienda_id, familia_id, fecha)
);
CREATE TABLE IF NOT EXISTS reentrenamientos (
    id TEXT PRIMARY KEY,
    solicitado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
    estado TEXT NOT NULL,
    promovido BOOLEAN,
    version TEXT,
    motivo TEXT,
    metricas JSONB,
    terminado_en TIMESTAMPTZ
);
-- Ventas conocidas: el histórico publicado más las observaciones (estas mandan si coinciden).
-- La promoción de un día observado es la de la última solicitud de pronóstico de ese día.
CREATE OR REPLACE VIEW ventas_diarias AS
SELECT h.tienda_id, h.familia_id, h.fecha, h.ventas, h.en_promocion
FROM historico h
WHERE NOT EXISTS (SELECT 1 FROM observaciones o WHERE o.tienda_id = h.tienda_id
                  AND o.familia_id = h.familia_id AND o.fecha = h.fecha)
UNION ALL
SELECT o.tienda_id, o.familia_id, o.fecha, o.ventas, p.en_promocion
FROM observaciones o
LEFT JOIN LATERAL (
    SELECT en_promocion FROM predicciones p
    WHERE p.tienda_id = o.tienda_id AND p.familia_id = o.familia_id AND p.fecha = o.fecha
    ORDER BY p.id DESC LIMIT 1
) p ON TRUE;
"""

COLUMNAS_VENTAS = ["tienda_id", "familia_id", "fecha", "ventas", "en_promocion"]


class RepositorioPostgres:
    def __init__(self, url: str, timeout: int = 5):
        self.url = url
        self.timeout = timeout

    @contextmanager
    def conexion(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.url, connect_timeout=self.timeout) as con:
            yield con

    # ----------------------------------------------------------------- infraestructura
    def crear_esquema(self) -> None:
        with self.conexion() as con:
            # Evita carreras si API, bootstrap y Airflow crean el esquema a la vez.
            con.execute("SELECT pg_advisory_xact_lock(4242)")
            con.execute(ESQUEMA)

    def ping(self) -> None:
        with self.conexion() as con:
            con.execute("SELECT 1")

    def metadato(self, clave: str) -> str | None:
        with self.conexion() as con:
            fila = con.execute("SELECT valor FROM metadatos WHERE clave = %s", (clave,)).fetchone()
            return fila[0] if fila else None

    def cargar_historico(self, ruta_csv: Path) -> int:
        """Carga historico_ventas.csv una sola vez (idempotente). Devuelve filas cargadas."""
        huella = hashlib.sha256(Path(ruta_csv).read_bytes()).hexdigest()
        with self.conexion() as con:
            con.execute("SELECT pg_advisory_xact_lock(4243)")
            fila = con.execute("SELECT valor FROM metadatos WHERE clave = 'historico_sha256'"
                               ).fetchone()
            if fila:
                if fila[0] != huella:
                    log.warning("historico_ventas.csv cambió desde la carga inicial; se conserva "
                                "la versión cargada")
                return 0
            df = pd.read_csv(ruta_csv)
            with con.cursor() as cur:
                with cur.copy("COPY historico (fecha, tienda_id, familia_id, ventas, en_promocion)"
                              " FROM STDIN") as copia:
                    for r in df[["fecha", "tienda_id", "familia_id", "ventas",
                                 "en_promocion"]].itertuples(index=False):
                        copia.write_row(tuple(r))
            con.execute("INSERT INTO metadatos (clave, valor) VALUES ('historico_sha256', %s)",
                        (huella,))
            return len(df)

    # ----------------------------------------------------------------------- lecturas
    def historial_tienda(self, tienda_id: int, fecha: dt.date, dias: int = 60) -> pd.DataFrame:
        """Ventas de la tienda en los `dias` previos al último dato anterior a `fecha`."""
        sql = """
            WITH ultimo AS (SELECT max(fecha) AS f FROM ventas_diarias
                            WHERE tienda_id = %(t)s AND fecha < %(f)s)
            SELECT tienda_id, familia_id, fecha, ventas, en_promocion FROM ventas_diarias, ultimo
            WHERE tienda_id = %(t)s AND fecha < %(f)s AND fecha > ultimo.f - %(d)s
        """
        with self.conexion() as con:
            filas = con.execute(sql, {"t": tienda_id, "f": fecha, "d": dias}).fetchall()
        return pd.DataFrame(filas, columns=COLUMNAS_VENTAS)

    def ventas_entrenamiento(self) -> pd.DataFrame:
        with self.conexion() as con:
            filas = con.execute(f"SELECT {', '.join(COLUMNAS_VENTAS)} FROM ventas_diarias"
                                ).fetchall()
        df = pd.DataFrame(filas, columns=COLUMNAS_VENTAS)
        df["ventas"] = df["ventas"].astype(float)
        df["en_promocion"] = pd.to_numeric(df["en_promocion"], errors="coerce")
        return df

    def conteo_observaciones(self) -> int:
        with self.conexion() as con:
            return con.execute("SELECT count(*) FROM observaciones").fetchone()[0]

    def maximos_historicos(self) -> dict[tuple[int, str], float]:
        with self.conexion() as con:
            filas = con.execute("SELECT tienda_id, familia_id, max(ventas) FROM historico "
                                "GROUP BY 1, 2").fetchall()
        return {(t, f): float(m) for t, f, m in filas}

    # ----------------------------------------------------------------------- escrituras
    def registrar_solicitud(self, solicitud_id: str, endpoint: str, cuerpo: Any, codigo: int,
                            tienda_id: int | None = None, fecha: dt.date | None = None,
                            calidad: dict | None = None, detalle: str | None = None,
                            version: str | None = None,
                            predicciones: list[dict] | None = None,
                            con: psycopg.Connection | None = None) -> None:
        def _escribir(c: psycopg.Connection) -> None:
            c.execute(
                "INSERT INTO solicitudes (id, endpoint, cuerpo, tienda_id, fecha, codigo_http, "
                "estado_calidad, detalle, version_modelo) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (solicitud_id, endpoint, Jsonb(_jsonable(cuerpo)), tienda_id, fecha, codigo,
                 (calidad or {}).get("estado"), detalle, version))
            obs = (calidad or {}).get("observaciones", [])
            if obs:
                with c.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO correcciones (solicitud_id, campo, accion, detalle) "
                        "VALUES (%s,%s,%s,%s)",
                        [(solicitud_id, o["campo"], o["accion"], o["detalle"]) for o in obs])
            if predicciones:
                with c.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO predicciones (solicitud_id, tienda_id, fecha, familia_id, "
                        "en_promocion, venta_esperada, cantidad_sugerida, version_modelo) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                        [(solicitud_id, tienda_id, fecha, p["familia_id"], p["en_promocion"],
                          p["venta_esperada"], p["cantidad_sugerida"], version)
                         for p in predicciones])

        if con is not None:
            _escribir(con)
        else:
            with self.conexion() as c:
                _escribir(c)

    def registrar_observacion(self, solicitud_id: str, cuerpo: Any, tienda_id: int,
                              fecha: dt.date, ventas: list[dict], bitacora) -> int:
        """Upsert idempotente. Devuelve registros nuevos o modificados y anota en la bitácora
        los valores que reemplazan a otros ya registrados."""
        with self.conexion() as con:
            familias = [v["familia_id"] for v in ventas]
            previos = dict(con.execute(
                "SELECT familia_id, ventas FROM observaciones WHERE tienda_id = %s AND fecha = %s "
                "AND familia_id = ANY(%s) FOR UPDATE", (tienda_id, fecha, familias)).fetchall())
            cambios = [v for v in ventas if previos.get(v["familia_id"]) != v["ventas"]]
            for v in cambios:
                if v["familia_id"] in previos:
                    bitacora.advertir(
                        "ventas", f"'{v['familia_id']}' ya estaba registrada con "
                                  f"{previos[v['familia_id']]:g}; se reemplazó por {v['ventas']:g}")
            self.registrar_solicitud(solicitud_id, "/v1/observaciones", cuerpo, 200, tienda_id,
                                     fecha, bitacora.calidad(), con=con)
            if cambios:
                with con.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO observaciones (tienda_id, familia_id, fecha, ventas, "
                        "solicitud_id) VALUES (%s,%s,%s,%s,%s) "
                        "ON CONFLICT (tienda_id, familia_id, fecha) DO UPDATE SET "
                        "ventas = EXCLUDED.ventas, solicitud_id = EXCLUDED.solicitud_id, "
                        "actualizado_en = now()",
                        [(tienda_id, v["familia_id"], fecha, v["ventas"], solicitud_id)
                         for v in cambios])
            return len(cambios)

    # ------------------------------------------------------------------ reentrenamientos
    def crear_reentrenamiento(self, id_: str) -> None:
        with self.conexion() as con:
            con.execute("INSERT INTO reentrenamientos (id, estado) VALUES (%s, 'en_curso') "
                        "ON CONFLICT (id) DO NOTHING", (id_,))

    def borrar_reentrenamiento(self, id_: str) -> None:
        with self.conexion() as con:
            con.execute("DELETE FROM reentrenamientos WHERE id = %s", (id_,))

    def obtener_reentrenamiento(self, id_: str) -> dict | None:
        with self.conexion() as con:
            fila = con.execute("SELECT id, estado, promovido, version, motivo FROM "
                               "reentrenamientos WHERE id = %s", (id_,)).fetchone()
        if not fila:
            return None
        return dict(zip(["id", "estado", "promovido", "version", "motivo"], fila))

    def finalizar_reentrenamiento(self, id_: str, estado: str, promovido: bool,
                                  version: str | None, motivo: str,
                                  metricas: dict | None = None) -> None:
        with self.conexion() as con:
            con.execute(
                "INSERT INTO reentrenamientos (id, estado, promovido, version, motivo, metricas, "
                "terminado_en) VALUES (%s,%s,%s,%s,%s,%s, now()) ON CONFLICT (id) DO UPDATE SET "
                "estado = EXCLUDED.estado, promovido = EXCLUDED.promovido, "
                "version = EXCLUDED.version, motivo = EXCLUDED.motivo, "
                "metricas = EXCLUDED.metricas, terminado_en = now()",
                (id_, estado, promovido, version, motivo, Jsonb(_jsonable(metricas or {}))))


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))
