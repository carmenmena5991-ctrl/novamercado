"""Modelos Pydantic del contrato (Anexo del enunciado).

Las solicitudes llegan crudas y pasan primero por `limpieza`; el resultado canónico se
valida aquí en modo estricto. Las respuestas se validan con `response_model`.
"""
from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_serializer

Estado = Literal["ok", "corregido", "advertencia"]
Accion = Literal["reparar", "advertir"]


class _Estricto(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


# ---------------------------------------------------------------- solicitudes canónicas
class FamiliaPromocion(_Estricto):
    familia_id: str
    en_promocion: NonNegativeInt


class SolicitudPrediccion(_Estricto):
    tienda_id: int
    fecha: dt.date
    familias: list[FamiliaPromocion] = Field(min_length=1)


class VentaFamilia(_Estricto):
    familia_id: str
    ventas: float = Field(ge=0)


class SolicitudObservacion(_Estricto):
    tienda_id: int
    fecha: dt.date
    ventas: list[VentaFamilia] = Field(min_length=1)


# ---------------------------------------------------------------------------- respuestas
class ObservacionCalidad(BaseModel):
    campo: str
    accion: Accion
    detalle: str


class Calidad(BaseModel):
    estado: Estado
    observaciones: list[ObservacionCalidad]


class PrediccionFamilia(BaseModel):
    familia_id: str
    venta_esperada: float
    cantidad_sugerida: NonNegativeInt


class RespuestaPrediccion(BaseModel):
    tienda_id: int
    fecha: dt.date
    version_modelo: str
    predicciones: list[PrediccionFamilia]
    calidad: Calidad

    @field_serializer("fecha")
    def _fecha(self, f: dt.date) -> str:
        return f.isoformat()


class RespuestaObservacion(BaseModel):
    registros_guardados: NonNegativeInt
    calidad: Calidad


class RespuestaReentrenarInicio(BaseModel):
    id: str


class RespuestaReentrenarEstado(BaseModel):
    model_config = ConfigDict(extra="allow")
    estado: Literal["en_curso", "exitoso", "fallido"]
    promovido: bool | None = None
    version: str | None = None
    motivo: str | None = None


class RespuestaModelo(BaseModel):
    nombre: str
    alias: str
    version: str


class RespuestaHealth(BaseModel):
    model_config = ConfigDict(extra="allow")
    estado: Literal["ok", "degradado"]
    motivo: str | None = None


# Ejemplo para la documentación OpenAPI (el cuerpo se recibe crudo).
EJEMPLO_PREDICCION = {
    "tienda_id": 5,
    "fecha": "2025-11-17",
    "familias": [{"familia_id": "abarrotes", "en_promocion": 0},
                 {"familia_id": "bebidas", "en_promocion": 8}],
}
EJEMPLO_OBSERVACION = {
    "tienda_id": 5,
    "fecha": "2025-11-17",
    "ventas": [{"familia_id": "abarrotes", "ventas": 994}],
}
