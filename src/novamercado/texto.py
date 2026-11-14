"""Normalización de texto compartida por catálogos y limpieza."""
from __future__ import annotations

import re
import unicodedata


def sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    """'  Frutas y Verduras ' -> 'frutas_y_verduras'; 'LÁCTEOS' -> 'lacteos'."""
    t = sin_acentos(str(texto)).lower().strip()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    return t.strip("_")
