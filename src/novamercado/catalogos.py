"""Catálogos publicados: tiendas, familias, costos y calendario de feriados."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .texto import normalizar


@dataclass
class Catalogos:
    tiendas: pd.DataFrame  # índice tienda_id; columnas nombre, ciudad, tipo, fecha_apertura
    familias: pd.DataFrame  # índice familia_id; columna nombre
    costos: pd.DataFrame  # índice familia_id; costo_faltante, costo_sobrante
    feriados: pd.DataFrame  # fecha, nombre, tipo, ambito
    _alias_familia: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.tiendas = self.tiendas.copy()
        self.tiendas["fecha_apertura"] = pd.to_datetime(self.tiendas["fecha_apertura"]).dt.date
        self.feriados = self.feriados.copy()
        self.feriados["fecha"] = pd.to_datetime(self.feriados["fecha"]).dt.date
        # Formas aceptadas de cada familia: id y nombre normalizados, y el nombre sin "y".
        for fid, fila in self.familias.iterrows():
            nombre = normalizar(fila["nombre"])
            for forma in {normalizar(fid), nombre, nombre.replace("_y_", "_")}:
                self._alias_familia[forma] = fid

    # --- Familias -----------------------------------------------------------------
    @property
    def ids_familia(self) -> list[str]:
        return list(self.familias.index)

    def familia_exacta(self, normalizado: str) -> str | None:
        return self._alias_familia.get(normalizado)

    @property
    def formas_familia(self) -> dict[str, str]:
        return dict(self._alias_familia)

    # --- Tiendas ------------------------------------------------------------------
    def existe_tienda(self, tienda_id: int) -> bool:
        return tienda_id in self.tiendas.index

    def apertura(self, tienda_id: int) -> dt.date:
        return self.tiendas.loc[tienda_id, "fecha_apertura"]

    def familias_de_tienda(self, tienda_id: int) -> list[str]:
        # Todas las tiendas venden las 12 familias en el histórico publicado.
        return self.ids_familia

    # --- Calendario ---------------------------------------------------------------
    @property
    def inicio_calendario(self) -> dt.date:
        return dt.date(min(self.feriados["fecha"]).year, 1, 1)

    @property
    def fin_calendario(self) -> dt.date:
        """Último día cubierto por feriados.csv (31 de diciembre del último año)."""
        return dt.date(max(self.feriados["fecha"]).year, 12, 31)

    def costo(self, familia_id: str) -> tuple[float, float]:
        fila = self.costos.loc[familia_id]
        return float(fila["costo_faltante"]), float(fila["costo_sobrante"])


def cargar_catalogos(datos_dir: Path) -> Catalogos:
    datos_dir = Path(datos_dir)
    return Catalogos(
        tiendas=pd.read_csv(datos_dir / "tiendas.csv").set_index("tienda_id"),
        familias=pd.read_csv(datos_dir / "familias.csv").set_index("familia_id"),
        costos=pd.read_csv(datos_dir / "costos.csv").set_index("familia_id"),
        feriados=pd.read_csv(datos_dir / "feriados.csv"),
    )
