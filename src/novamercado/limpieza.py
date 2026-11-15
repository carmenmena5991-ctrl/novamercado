"""Defensa del contrato: convierte un cuerpo JSON crudo en una solicitud canónica.

Cada problema se resuelve con una de tres políticas (DECISIONES.md, sección 1):

* reparar   - la corrección es inequívoca; se informa en `calidad`.
* advertir  - se responde con un supuesto explícito; se informa en `calidad`.
* rechazar  - se lanza `Rechazo` y el endpoint responde 422 con `detail`.

Las reglas son generales (por tipo de dato y de estructura), no por ejemplo del catálogo.
Este módulo no conoce FastAPI ni la base de datos: recibe Python puro y devuelve Python puro.
"""
from __future__ import annotations

import datetime as dt
import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .catalogos import Catalogos
from .texto import normalizar

REPARAR = "reparar"
ADVERTIR = "advertir"

# Máximo de días entre el último dato de ventas y la fecha pedida sin advertir.
DIAS_SIN_DATOS_TOLERADOS = 7

# Productos en promoción por familia por encima de este valor se consideran inverosímiles
# (el histórico publicado no pasa de unas decenas): se acotan con advertencia.
PROMOCION_MAXIMA = 1000


class Rechazo(Exception):
    """La solicitud no se puede atender sin inventar datos: se responde 422."""

    def __init__(self, detalle: str):
        super().__init__(detalle)
        self.detalle = detalle


@dataclass
class Bitacora:
    observaciones: list[dict[str, str]] = field(default_factory=list)

    def reparar(self, campo: str, detalle: str) -> None:
        self.observaciones.append({"campo": campo, "accion": REPARAR, "detalle": detalle})

    def advertir(self, campo: str, detalle: str) -> None:
        self.observaciones.append({"campo": campo, "accion": ADVERTIR, "detalle": detalle})

    @property
    def estado(self) -> str:
        acciones = {o["accion"] for o in self.observaciones}
        if ADVERTIR in acciones:
            return "advertencia"
        if REPARAR in acciones:
            return "corregido"
        return "ok"

    def calidad(self) -> dict[str, Any]:
        return {"estado": self.estado, "observaciones": list(self.observaciones)}


# --------------------------------------------------------------------------------------
# Nombres de campo
# --------------------------------------------------------------------------------------
# Sinónimos conocidos (ya normalizados). Además se aplica coincidencia aproximada.
ALIAS_CAMPOS: dict[str, set[str]] = {
    "tienda_id": {"tienda", "id_tienda", "tiendaid", "store", "store_id", "codigo_tienda",
                  "tienda_codigo", "cod_tienda", "sucursal", "sucursal_id", "id_sucursal"},
    "fecha": {"fecha_pronostico", "fecha_prediccion", "fecha_venta", "fecha_ventas", "dia",
              "date", "fecha_observacion", "fecha_solicitud"},
    "familias": {"familia", "lista_familias", "productos", "categorias", "items", "detalle",
                 "lineas", "promociones"},
    "familia_id": {"familia", "id_familia", "familiaid", "familia_nombre", "nombre_familia",
                   "categoria", "categoria_id", "family", "family_id", "nombre"},
    "en_promocion": {"promocion", "promociones", "en_promo", "promo", "promos",
                     "productos_en_promocion", "cantidad_promocion", "promotion",
                     "on_promotion", "onpromotion", "en_promociones"},
    "ventas": {"venta", "ventas_reales", "cantidad", "unidades", "total", "valor", "sales",
               "unidades_vendidas", "observaciones", "datos", "familias", "lineas", "items",
               "detalle"},
}


def _destino_campo(clave: str, permitidos: list[str]) -> tuple[str | None, bool]:
    """Devuelve (campo canónico, es_exacto) para una clave recibida."""
    norm = normalizar(clave)
    if clave in permitidos:
        return clave, True
    if norm in permitidos:
        return norm, False
    for canonico in permitidos:
        if norm in ALIAS_CAMPOS.get(canonico, set()):
            return canonico, False
    # Coincidencia aproximada con el nombre canónico (errores de tipeo: "tienda_di").
    cercanos = difflib.get_close_matches(norm, permitidos, n=2, cutoff=0.8)
    if len(cercanos) == 1:
        return cercanos[0], False
    return None, False


def mapear_campos(obj: dict, permitidos: list[str], prefijo: str, bit: Bitacora) -> dict:
    """Renombra claves al contrato. Exacto > sinónimo; conflictos se rechazan."""
    resultado: dict[str, Any] = {}
    origen: dict[str, str] = {}
    exactos = {k for k in obj if k in permitidos}
    for clave, valor in obj.items():
        destino, exacto = _destino_campo(str(clave), permitidos)
        ruta = f"{prefijo}{clave}"
        if destino is None:
            bit.advertir(ruta, f"campo '{clave}' no pertenece al contrato; se ignoró")
            continue
        if not exacto and destino in exactos:
            # Llega el campo correcto y además un sinónimo: manda el del contrato.
            if valor != obj[destino]:
                bit.advertir(ruta, f"campo '{clave}' duplica a '{destino}' con otro valor; "
                                   f"se usó '{destino}'")
            continue
        if destino in resultado:
            if resultado[destino] != valor:
                raise Rechazo(f"los campos '{origen[destino]}' y '{clave}' indican "
                              f"'{destino}' con valores distintos")
            continue
        resultado[destino] = valor
        origen[destino] = str(clave)
        if not exacto:
            bit.reparar(f"{prefijo}{destino}", f"el campo '{clave}' se interpretó como '{destino}'")
    return resultado


# --------------------------------------------------------------------------------------
# Números
# --------------------------------------------------------------------------------------
@dataclass
class Numero:
    valor: float
    nota: str | None = None  # descripción de la reparación, si la hubo
    ambiguo: bool = False  # True -> advertir en lugar de reparar


_ENTERO = re.compile(r"^[+-]?\d+$")
_DEC_PUNTO = re.compile(r"^[+-]?\d+\.\d+$")
_MILES_COMA = re.compile(r"^[+-]?\d{1,3}(,\d{3})+$")
_MILES_PUNTO = re.compile(r"^[+-]?\d{1,3}(\.\d{3}){2,}$")
_MILES_PUNTO_UNO = re.compile(r"^[+-]?\d{1,3}\.\d{3}$")
_DEC_COMA = re.compile(r"^[+-]?\d+,\d{1,2}$")
_MILES_COMA_DEC_PUNTO = re.compile(r"^[+-]?\d{1,3}(,\d{3})+\.\d+$")
_MILES_PUNTO_DEC_COMA = re.compile(r"^[+-]?\d{1,3}(\.\d{3})+,\d+$")


def parsear_numero(valor: Any) -> Numero | None:
    """Interpreta números enviados como texto o con formatos regionales.

    Devuelve None si el valor no se puede interpretar como número.
    """
    if isinstance(valor, bool) or valor is None:
        return None
    if isinstance(valor, int):
        return Numero(float(valor))
    if isinstance(valor, float):
        if not math.isfinite(valor):
            return None
        if valor.is_integer():
            return Numero(valor, f"{valor!r} se interpretó como {int(valor)}")
        return Numero(valor)
    if not isinstance(valor, str):
        return None
    t = valor.strip().replace(" ", "").replace(" ", "")
    if not t:
        return None
    original = repr(valor)
    if _ENTERO.match(t):
        return Numero(float(int(t)), f"{original} (texto) se interpretó como {int(t)}")
    if _MILES_COMA.match(t):
        n = float(t.replace(",", ""))
        return Numero(n, f"{original} se interpretó como {_fmt(n)} (coma como separador de miles)")
    if _MILES_PUNTO.match(t):
        n = float(t.replace(".", ""))
        return Numero(n, f"{original} se interpretó como {_fmt(n)} (punto como separador de miles)")
    if _MILES_PUNTO_UNO.match(t):
        # "1.234": ¿mil doscientos treinta y cuatro o uno punto dos? En unidades enteras lo
        # plausible es el separador de miles, pero no es inequívoco.
        n = float(t.replace(".", ""))
        return Numero(n, f"{original} es ambiguo; se supuso {_fmt(n)} (punto como separador "
                         f"de miles)", ambiguo=True)
    if _DEC_PUNTO.match(t):
        n = float(t)
        return Numero(n, f"{original} (texto) se interpretó como {_fmt(n)}")
    if _DEC_COMA.match(t):
        n = float(t.replace(",", "."))
        return Numero(n, f"{original} se interpretó como {_fmt(n)} (coma decimal)")
    if _MILES_COMA_DEC_PUNTO.match(t):
        n = float(t.replace(",", ""))
        return Numero(n, f"{original} se interpretó como {_fmt(n)}")
    if _MILES_PUNTO_DEC_COMA.match(t):
        n = float(t.replace(".", "").replace(",", "."))
        return Numero(n, f"{original} se interpretó como {_fmt(n)}")
    return None


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


# --------------------------------------------------------------------------------------
# Fechas
# --------------------------------------------------------------------------------------
@dataclass
class Fecha:
    valor: dt.date
    nota: str | None = None
    ambigua: bool = False


MESES = {
    "ene": 1, "enero": 1, "feb": 2, "febrero": 2, "mar": 3, "marzo": 3, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5, "jun": 6, "junio": 6, "jul": 7, "julio": 7, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "set": 9, "septiembre": 9, "setiembre": 9, "oct": 10, "octubre": 10,
    "nov": 11, "noviembre": 11, "dic": 12, "diciembre": 12,
    "jan": 1, "january": 1, "february": 2, "march": 3, "apr": 4, "april": 4, "june": 6,
    "july": 7, "aug": 8, "august": 8, "september": 9, "october": 10, "november": 11,
    "dec": 12, "december": 12,
}

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_HORA = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")
_AMD = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
_COMPACTA = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_DMA = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$")
_TEXTO = re.compile(r"^(\d{1,2})(?:\s+de\s+|[\s\-/.]+)([a-záéíóú]+)\.?(?:\s+de\s+|[\s\-/.,]+)(\d{4})$")


def _crear_fecha(a: int, m: int, d: int, original: str) -> dt.date:
    try:
        return dt.date(a, m, d)
    except ValueError:
        raise Rechazo(f"fecha {original!r} no existe en el calendario") from None


def parsear_fecha(valor: Any) -> Fecha:
    """Interpreta la fecha; lanza Rechazo si no se puede determinar."""
    if isinstance(valor, bool) or valor is None:
        raise Rechazo("la fecha es obligatoria y debe tener formato AAAA-MM-DD")
    if isinstance(valor, int):
        valor = str(valor)
    if not isinstance(valor, str):
        raise Rechazo(f"fecha {valor!r} no es texto con formato AAAA-MM-DD")
    t = valor.strip()
    if m := _ISO.match(t):
        f = _crear_fecha(int(m[1]), int(m[2]), int(m[3]), valor)
        return Fecha(f, None if t == valor else f"{valor!r} se interpretó como {f.isoformat()}")
    if m := _ISO_HORA.match(t):
        f = _crear_fecha(int(m[1]), int(m[2]), int(m[3]), valor)
        return Fecha(f, f"{valor!r} se interpretó como {f.isoformat()} (se descartó la hora)")
    if m := (_AMD.match(t) or _COMPACTA.match(t)):
        f = _crear_fecha(int(m[1]), int(m[2]), int(m[3]), valor)
        return Fecha(f, f"{valor!r} se interpretó como {f.isoformat()} (año-mes-día)")
    if m := _DMA.match(t):
        a, b, anio = int(m[1]), int(m[2]), int(m[3])
        if a > 12 and b <= 12:
            f = _crear_fecha(anio, b, a, valor)
            return Fecha(f, f"{valor!r} se interpretó como {f.isoformat()} (día/mes/año)")
        if b > 12 and a <= 12:
            f = _crear_fecha(anio, a, b, valor)
            return Fecha(f, f"{valor!r} se interpretó como {f.isoformat()} (mes/día/año)")
        if a == b:
            f = _crear_fecha(anio, b, a, valor)
            return Fecha(f, f"{valor!r} se interpretó como {f.isoformat()}")
        if a > 12 and b > 12:
            raise Rechazo(f"fecha {valor!r} no es válida")
        f = _crear_fecha(anio, b, a, valor)
        return Fecha(f, f"{valor!r} es ambigua; se supuso día/mes/año (convención local): "
                        f"{f.isoformat()}", ambigua=True)
    if m := _TEXTO.match(t.lower()):
        from .texto import sin_acentos
        mes = MESES.get(sin_acentos(m[2]))
        if mes:
            f = _crear_fecha(int(m[3]), mes, int(m[1]), valor)
            return Fecha(f, f"{valor!r} se interpretó como {f.isoformat()}")
    raise Rechazo(f"fecha {valor!r} no tiene un formato reconocible; use AAAA-MM-DD")


def aplicar_fecha(valor: Any, tienda_id: int, cat: Catalogos, bit: Bitacora) -> dt.date:
    f = parsear_fecha(valor)
    if f.nota:
        (bit.advertir if f.ambigua else bit.reparar)("fecha", f.nota)
    fecha = f.valor
    # Validez sin reloj del sistema: dentro del calendario publicado y con la tienda abierta.
    if fecha > cat.fin_calendario:
        raise Rechazo(f"la fecha {fecha.isoformat()} es posterior al calendario de feriados "
                      f"publicado (hasta {cat.fin_calendario.isoformat()})")
    if fecha < cat.inicio_calendario:
        raise Rechazo(f"la fecha {fecha.isoformat()} es anterior al periodo con datos "
                      f"({cat.inicio_calendario.isoformat()})")
    if fecha < cat.apertura(tienda_id):
        raise Rechazo(f"la tienda {tienda_id} abrió el {cat.apertura(tienda_id).isoformat()}; "
                      f"no hay ventas para {fecha.isoformat()}")
    return fecha


# --------------------------------------------------------------------------------------
# Tienda
# --------------------------------------------------------------------------------------
def aplicar_tienda(valor: Any, cat: Catalogos, bit: Bitacora) -> int:
    tid: int | None = None
    nota: str | None = None
    if isinstance(valor, bool) or valor is None:
        raise Rechazo("tienda_id es obligatorio y debe ser un entero del catálogo de tiendas")
    if isinstance(valor, int):
        tid = valor
    elif isinstance(valor, float) and valor.is_integer():
        tid, nota = int(valor), f"{valor!r} se interpretó como {int(valor)}"
    elif isinstance(valor, str):
        t = valor.strip()
        numeros = re.findall(r"\d+", t)
        resto = re.sub(r"\d+", "", t)
        if re.fullmatch(r"\d+", t):
            tid = int(t)
        elif len(numeros) == 1 and re.fullmatch(r"[A-Za-z_#\-\s.:]*", resto) and \
                normalizar(resto) in {"", "t", "tienda", "tda", "id", "no", "num", "store", "s",
                                      "sucursal", "tienda_no", "tienda_num"}:
            tid = int(numeros[0])
        else:
            # Nombre de la tienda ("NovaMercado Mixco" o "Mixco").
            norm = normalizar(t)
            for i, fila in cat.tiendas.iterrows():
                nombre = normalizar(fila["nombre"])
                if norm in {nombre, nombre.removeprefix("novamercado_")}:
                    tid = int(i)
        if tid is not None:
            nota = f"{valor!r} se interpretó como tienda {tid}"
    if tid is None:
        raise Rechazo(f"tienda_id {valor!r} no identifica una tienda")
    if not cat.existe_tienda(tid):
        raise Rechazo(f"la tienda {tid} no existe en el catálogo de tiendas")
    if nota:
        bit.reparar("tienda_id", nota)
    return tid


# --------------------------------------------------------------------------------------
# Familia
# --------------------------------------------------------------------------------------
def resolver_familia(valor: Any, cat: Catalogos) -> tuple[str | None, bool]:
    """Devuelve (familia_id, es_aproximada). None si no se reconoce."""
    if not isinstance(valor, str) or not valor.strip():
        return None, False
    norm = normalizar(valor)
    exacta = cat.familia_exacta(norm)
    if exacta:
        return exacta, False
    formas = cat.formas_familia
    cercanos = difflib.get_close_matches(norm, list(formas), n=3, cutoff=0.8)
    candidatos = {formas[c] for c in cercanos}
    if len(candidatos) == 1:
        return candidatos.pop(), True
    return None, False


# --------------------------------------------------------------------------------------
# Lista de elementos (familias / ventas)
# --------------------------------------------------------------------------------------
def _como_lista(valor: Any, campo: str, clave_valor: str, bit: Bitacora) -> list:
    if isinstance(valor, list):
        return valor
    if isinstance(valor, dict) and valor and all(isinstance(k, str) for k in valor):
        # {"abarrotes": 3, ...} -> [{"familia_id": "abarrotes", clave_valor: 3}, ...]
        bit.reparar(campo, f"'{campo}' llegó como objeto; se convirtió a lista de elementos")
        return [{"familia_id": k, clave_valor: v} for k, v in valor.items()]
    raise Rechazo(f"'{campo}' debe ser una lista de elementos con familia_id")


def _resolver_elementos(lista: list, campo: str, clave_valor: str, cat: Catalogos,
                        bit: Bitacora) -> list[tuple[int, str, Any, bool]]:
    """Normaliza cada elemento. Devuelve (índice, familia_id, valor crudo, valor_presente)."""
    salida = []
    for i, elem in enumerate(lista):
        ruta = f"{campo}[{i}]"
        if isinstance(elem, str):
            elem = {"familia_id": elem}
        if not isinstance(elem, dict):
            bit.advertir(ruta, f"elemento {elem!r} no es un objeto; se ignoró")
            continue
        elem = mapear_campos(elem, ["familia_id", clave_valor], f"{ruta}.", bit)
        crudo = elem.get("familia_id")
        fid, aprox = resolver_familia(crudo, cat)
        if fid is None:
            bit.advertir(f"{ruta}.familia_id", f"familia {crudo!r} no existe en el catálogo; "
                                               f"se ignoró el elemento")
            continue
        if crudo != fid:
            if aprox:
                bit.advertir(f"{ruta}.familia_id", f"'{crudo}' se supuso '{fid}' (nombre parecido)")
            else:
                bit.reparar(f"{ruta}.familia_id", f"'{crudo}' se interpretó como '{fid}'")
        salida.append((i, fid, elem.get(clave_valor), clave_valor in elem))
    return salida


def _agrupar_duplicados(elems: list[tuple[int, str, Any]], campo: str, bit: Bitacora,
                        conflicto: str) -> dict[str, tuple[int, Any]]:
    """Agrupa por familia. conflicto='ultimo' usa el último valor; 'omitir' descarta."""
    por_familia: dict[str, list[tuple[int, Any]]] = {}
    for i, fid, v in elems:
        por_familia.setdefault(fid, []).append((i, v))
    resultado: dict[str, tuple[int, Any]] = {}
    for fid, ocurrencias in por_familia.items():
        if len(ocurrencias) == 1:
            resultado[fid] = ocurrencias[0]
            continue
        indices = ", ".join(str(i) for i, _ in ocurrencias)
        valores = [v for _, v in ocurrencias]
        if all(v == valores[0] for v in valores):
            bit.reparar(campo, f"'{fid}' aparece {len(ocurrencias)} veces (posiciones {indices}) "
                               f"con el mismo valor; se usó una sola vez")
            resultado[fid] = ocurrencias[-1]
        elif conflicto == "ultimo":
            bit.advertir(campo, f"'{fid}' aparece {len(ocurrencias)} veces con valores distintos "
                                f"({valores}); se usó el último ({valores[-1]})")
            resultado[fid] = ocurrencias[-1]
        else:
            bit.advertir(campo, f"'{fid}' aparece {len(ocurrencias)} veces con valores distintos "
                                f"({valores}); no se registró por ser contradictorio")
    return resultado


def _exigir_objeto(cuerpo: Any) -> dict:
    if not isinstance(cuerpo, dict):
        raise Rechazo("el cuerpo debe ser un objeto JSON")
    return cuerpo


def _obligatorio(datos: dict, campo: str) -> Any:
    if campo not in datos or datos[campo] is None:
        raise Rechazo(f"falta el campo obligatorio '{campo}'")
    return datos[campo]


# --------------------------------------------------------------------------------------
# POST /v1/prediccion
# --------------------------------------------------------------------------------------
def limpiar_prediccion(cuerpo: Any, cat: Catalogos) -> tuple[dict, Bitacora]:
    bit = Bitacora()
    datos = mapear_campos(_exigir_objeto(cuerpo), ["tienda_id", "fecha", "familias"], "", bit)
    tienda = aplicar_tienda(_obligatorio(datos, "tienda_id"), cat, bit)
    fecha = aplicar_fecha(_obligatorio(datos, "fecha"), tienda, cat, bit)
    lista = _como_lista(_obligatorio(datos, "familias"), "familias", "en_promocion", bit)
    if not lista:
        raise Rechazo("'familias' está vacía: no hay familias que pronosticar")

    elementos = []
    for i, fid, crudo, presente in _resolver_elementos(lista, "familias", "en_promocion", cat, bit):
        elementos.append((i, fid, _promocion(crudo, presente, f"familias[{i}].en_promocion", bit)))
    if not elementos:
        raise Rechazo("ninguna familia de la solicitud existe en el catálogo")
    unicos = _agrupar_duplicados(elementos, "familias", bit, conflicto="ultimo")

    familias = [{"familia_id": fid, "en_promocion": v} for fid, (_, v) in
                sorted(unicos.items(), key=lambda kv: kv[1][0])]
    faltantes = [f for f in cat.familias_de_tienda(tienda) if f not in unicos]
    if faltantes:
        bit.advertir("familias", f"faltan {len(faltantes)} familias que la tienda vende "
                                 f"({', '.join(faltantes)}); se pronosticaron suponiendo "
                                 f"en_promocion = 0")
        familias += [{"familia_id": f, "en_promocion": 0} for f in faltantes]
    return {"tienda_id": tienda, "fecha": fecha, "familias": familias}, bit


def _promocion(crudo: Any, presente: bool, ruta: str, bit: Bitacora) -> int:
    if not presente or crudo is None:
        bit.advertir(ruta, "en_promocion no se envió; se supuso 0")
        return 0
    n = parsear_numero(crudo)
    if n is None:
        bit.advertir(ruta, f"en_promocion {crudo!r} no es numérico; se supuso 0")
        return 0
    if n.valor < 0:
        bit.advertir(ruta, f"en_promocion {crudo!r} es negativo; se supuso 0")
        return 0
    if n.valor > PROMOCION_MAXIMA:
        bit.advertir(ruta, f"en_promocion {crudo!r} es inverosímil (más de {PROMOCION_MAXIMA}); "
                           f"se acotó a {PROMOCION_MAXIMA}")
        return PROMOCION_MAXIMA
    if not float(n.valor).is_integer():
        redondeado = int(round(n.valor))
        bit.advertir(ruta, f"en_promocion {crudo!r} no es entero; se redondeó a {redondeado}")
        return redondeado
    if n.nota:
        (bit.advertir if n.ambiguo else bit.reparar)(ruta, n.nota)
    return int(n.valor)


# --------------------------------------------------------------------------------------
# POST /v1/observaciones
# --------------------------------------------------------------------------------------
def limpiar_observacion(cuerpo: Any, cat: Catalogos,
                        maximos: dict[tuple[int, str], float] | None = None,
                        factor_atipico: float = 10.0) -> tuple[dict, Bitacora]:
    """`maximos[(tienda, familia)]` es la venta máxima histórica; valores mayores a
    factor_atipico veces ese máximo se consideran errores de captura y no se registran."""
    bit = Bitacora()
    datos = mapear_campos(_exigir_objeto(cuerpo), ["tienda_id", "fecha", "ventas"], "", bit)
    tienda = aplicar_tienda(_obligatorio(datos, "tienda_id"), cat, bit)
    fecha = aplicar_fecha(_obligatorio(datos, "fecha"), tienda, cat, bit)
    lista = _como_lista(_obligatorio(datos, "ventas"), "ventas", "ventas", bit)
    if not lista:
        raise Rechazo("'ventas' está vacía: no hay ventas que registrar")

    elementos = []
    for i, fid, crudo, presente in _resolver_elementos(lista, "ventas", "ventas", cat, bit):
        ruta = f"ventas[{i}].ventas"
        v = _venta(crudo, presente, ruta, bit)
        if v is None:
            continue
        maximo = (maximos or {}).get((tienda, fid))
        if maximo and v > factor_atipico * maximo:
            bit.advertir(ruta, f"venta {_fmt(v)} supera {factor_atipico:g} veces el máximo "
                               f"histórico ({_fmt(maximo)}); no se registró")
            continue
        elementos.append((i, fid, v))
    unicos = _agrupar_duplicados(elementos, "ventas", bit, conflicto="omitir")
    if not unicos:
        raise Rechazo("ninguna venta de la observación es válida; no se registró nada")

    faltantes = [f for f in cat.familias_de_tienda(tienda) if f not in unicos]
    if faltantes:
        bit.advertir("ventas", f"no se recibieron ventas válidas de {len(faltantes)} familias "
                               f"({', '.join(faltantes)}); quedan sin registrar")
    ventas = [{"familia_id": fid, "ventas": v} for fid, (_, v) in
              sorted(unicos.items(), key=lambda kv: kv[1][0])]
    return {"tienda_id": tienda, "fecha": fecha, "ventas": ventas}, bit


def _venta(crudo: Any, presente: bool, ruta: str, bit: Bitacora) -> float | None:
    if not presente or crudo is None:
        bit.advertir(ruta, "no se envió el valor de ventas; no se registró")
        return None
    n = parsear_numero(crudo)
    if n is None:
        bit.advertir(ruta, f"ventas {crudo!r} no es numérico; no se registró")
        return None
    if n.valor < 0:
        bit.advertir(ruta, f"ventas {crudo!r} es negativo (posible devolución); no se registró "
                           f"para no inventar la venta real")
        return None
    if not float(n.valor).is_integer():
        redondeado = float(round(n.valor))
        bit.advertir(ruta, f"ventas {crudo!r} no es un número entero de unidades; se redondeó "
                           f"a {_fmt(redondeado)}")
        return redondeado
    if n.nota:
        (bit.advertir if n.ambiguo else bit.reparar)(ruta, n.nota)
    return n.valor
