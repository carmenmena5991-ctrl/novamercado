# Decisiones técnicas y de producto

Principio rector: **el servicio no se detiene por datos imperfectos, pero tampoco inventa
datos**. Cuando la corrección es inequívoca se repara; cuando hay que suponer algo se
responde y se dice qué se supuso; cuando responder obligaría a inventar, se rechaza.

## 1. Política ante problemas en los datos

Las reglas son **por tipo de problema**, no por ejemplo del catálogo, para cubrir casos no
vistos. Implementación: `src/novamercado/limpieza.py`; pruebas: `tests/test_limpieza.py`.

| Tipo de problema | Política | Regla general |
|---|---|---|
| Nombre de campo distinto (E07) | reparar | Se normaliza (minúsculas, sin acentos, `_`); se acepta sinónimo conocido o coincidencia aproximada única (≥ 0.8). Dos sinónimos con valores distintos → **rechazar**. |
| Campo que no es del contrato | advertir | Se ignora y se informa. |
| Campo obligatorio ausente o nulo | rechazar | No hay con qué pronosticar. |
| Cuerpo que no es objeto / JSON inválido | rechazar | 422 con `detail`. |
| Tienda como texto o con prefijo (E01) | reparar | `"5"`, `"T05"`, `"Tienda 5"`, `5.0`, nombre exacto de la tienda. |
| Tienda inexistente, ambigua (`"5-6"`), decimal (`5.5`), booleano | rechazar | No se puede suponer de qué tienda se trata. |
| Fecha en otro formato inequívoco (E04) | reparar | ISO con hora, `AAAA/MM/DD`, `AAAAMMDD`, `DD/MM/AAAA` con día > 12, `MM/DD/AAAA` con mes ≤ 12 < día, "17 de noviembre de 2025". |
| Fecha ambigua (`05/11/2025`) | advertir | Se supone día/mes/año (convención local). |
| Fecha inexistente o ilegible | rechazar | |
| Fecha fuera del calendario de `feriados.csv` o antes de la apertura de la tienda | rechazar | Validez **sin reloj del sistema**: el rango lo define el calendario publicado. |
| Último dato de ventas a más de 7 días de la fecha pedida | advertir | Se pronostica con el último nivel conocido. |
| Familia escrita de otra forma (E02) | reparar | Coincidencia exacta tras normalizar contra id y nombre (`LÁCTEOS`, `Frutas y Verduras`). |
| Familia con error de tipeo (`abarotes`) | advertir | Coincidencia aproximada única. |
| Familia desconocida | advertir | Se ignora ese elemento. |
| Faltan familias que la tienda vende (E05) | advertir | Se pronostican igual suponiendo `en_promocion = 0`. Alternativa descartada: responder solo las enviadas (la tienda necesita todas para su pedido). |
| Familia repetida, mismo valor | reparar | Se usa una vez. |
| Familia repetida, valores distintos (E06) | advertir | Se usa el **último** valor (corrección más reciente) y se informan todos. |
| Promoción como texto o `2.0` (E03) | reparar | |
| Promoción decimal, negativa, no numérica o ausente | advertir | Se redondea o se supone 0. |
| Lista como objeto `{"abarrotes": 3}` | reparar | Se convierte a lista. |
| Lista vacía / ninguna familia válida | rechazar | |
| Ventas con separador de miles o coma decimal (E10) | reparar | `"1,234"` → 1234, `"812,0"` → 812, `"1.234.567"`, `"1.234,5"`. |
| Número ambiguo (`"1.234"`) | advertir | Se supone separador de miles (en unidades enteras es lo plausible). |
| Ventas decimales | advertir | Se redondea a unidades. |
| Ventas negativas (E08), no numéricas o ausentes | advertir | **No se registra** ese registro: guardar 0 o el valor absoluto sería inventar la venta real. |
| Ventas > 10 × máximo histórico de la tienda-familia | advertir | No se registra (probable error de captura que contaminaría el reentrenamiento). |
| Familia repetida en una observación con valores distintos | advertir | No se registra ninguno (contradictorio). |
| Envío repetido de la misma observación (E09) | ok | Upsert idempotente: `registros_guardados = 0`. |
| Observación que cambia un valor ya registrado | advertir | Se reemplaza (la última manda) y se informa el valor previo. |

`calidad.estado`: `advertencia` si hay alguna acción `advertir`; si no, `corregido` si hay
alguna `reparar`; si no, `ok` sin observaciones. Rechazos: 422 con `detail`. Todas las
solicitudes (también las rechazadas) y cada corrección se guardan en la base del producto.

## 2. Información que usa el modelo al predecir la fecha F

- **Ventas anteriores a F** de la misma tienda y familia (histórico + observaciones):
  venta del último día, media de los últimos 7 y 28 días observados, media de los últimos
  4 mismos días de la semana, desviación de 28 días, tendencia (media 7 / media 28) y días
  desde el último dato. Nunca la venta de F (hay prueba que lo verifica).
- **Promoción de F** que llega en la solicitud, absoluta y relativa a las promociones de los
  28 días previos. Las observaciones no traen promoción: para entrenar con días observados
  se usa la de la última solicitud de pronóstico de ese día.
- **Calendario de F:** día de la semana, del mes, del año, mes, fin de mes, quincena,
  feriado/comercial/pago/medio día **nacional o de la ciudad de la tienda**, víspera y día
  posterior a feriado, días a Navidad.
- **Atributos fijos:** tienda, tipo de tienda y familia (categóricas).

El modelo predice el **cociente venta / media de 28 días** (ponderado por el nivel). Así es
independiente de la escala de la tienda y se adapta solo a cambios de nivel de la demanda:
si la demanda sube 30 %, las medias recientes suben y el pronóstico con ellas, sin esperar
a un reentrenamiento. Las ventanas cuentan días *observados*, para tolerar huecos.

**Modelos:** baseline = media de los últimos 4 mismos días de la semana;
modelo de ML = `HistGradientBoostingRegressor` de scikit-learn (rápido, maneja nulos y
categóricas, sin dependencias extra). En los últimos 28 días del histórico el ML reduce el
costo de negocio 30 % frente al baseline (91 771 vs 131 357) y el WAPE de 0.215 a 0.175.

## 3. Cantidad sugerida

El costo de cada familia es asimétrico (`costos.csv`): faltar licores cuesta 20 veces lo
que sobrar; sobrar pan cuesta 2.5 veces lo que faltar. La cantidad que minimiza el costo
esperado es un cuantil de la demanda (problema del *newsvendor*, cuantil crítico
c_faltante / (c_faltante + c_sobrante)), no la media.

Decisión: `cantidad_sugerida = redondeo(venta_esperada × k[familia, tipo de tienda])`, con
`k` elegido en una grilla 0.70–2.00 **minimizando directamente el costo de costos.csv**
sobre predicciones fuera de muestra (28 días de calibración). Es la versión empírica del
newsvendor: no supone distribución, incluye el efecto del redondeo y usa el error real del
modelo (el multiplicativo crece con el nivel, por eso es un factor y no una suma).
Resultado: abarrotes, licores o limpieza reciben k > 1; carnes, panadería y frutas y
verduras k ≈ 1 o menor. Reduce el costo ≈ 20 % frente a sugerir la venta esperada.

Alternativas descartadas: regresión cuantílica por familia (12 modelos, más lenta y frágil
con pocas filas por grupo); cuantil normal con σ estimada (supone simetría que no hay en
conteos pequeños).

## 4. Criterio de reemplazo del modelo

Ventana = últimos 56 días con ventas conocidas: 28 de calibración + 28 de **prueba**.
1. Se entrenan el baseline y el ML con los datos anteriores a la ventana, se calibran sus
   factores en calibración y se mide el **costo de negocio en prueba**. Gana el de menor costo.
2. El modelo vigente (`pronostico_ventas@campeon`) se evalúa en los mismos 28 días de prueba.
3. **Se promueve** si `costo_candidato < costo_vigente × (1 − 0.01)`, o si no hay vigente o
   este no puede evaluarse. El 1 % evita cambiar de modelo por ruido.
4. Se registra **exactamente el modelo evaluado** (no se reentrena después con la ventana).
   Así el vigente nunca vio los días de prueba de reentrenamientos posteriores y la
   comparación es fuera de muestra para ambos. (Una primera versión reentrenaba con todos
   los datos: el vigente se evaluaba sobre días que conocía y casi nunca perdía.)
   El costo de no usar los últimos 56 días para ajustar los árboles es bajo: esos días sí
   entran en las variables de nivel al predecir y en la calibración de `k`.

Toda versión candidata queda registrada en el Model Registry con etiquetas `promovido`,
`costo_validacion` y `tipo`, y el run guarda parámetros, métricas del baseline, del ML,
del candidato y del vigente (linaje completo). El alias se mueve solo al final: si cualquier
paso falla, el vigente no cambia y la ejecución queda `fallido`. El `motivo` de
`GET /admin/reentrenar/{id}` explica la decisión con ambos costos.

## 5. Arquitectura y operación

- **Base del producto separada** (`db`, Postgres 16.10) de la de Airflow (`airflow-db`).
  Tablas: `historico`, `observaciones` (PK tienda+familia+fecha → idempotencia),
  `solicitudes` (cuerpo original, código HTTP, estado de calidad), `correcciones`,
  `predicciones` (con `version_modelo`), `reentrenamientos`, `metadatos`. La vista
  `ventas_diarias` une histórico y observaciones (estas mandan).
- **Recarga en caliente:** un hilo de la API consulta cada 15 s el alias en MLflow; si cambió
  descarga la versión y cambia la referencia atómicamente (promoción visible en ≤ ~30 s,
  límite exigido 2 min). Guarda una copia local (`cache_modelo`) para arrancar sin MLflow.
- **Salud:** el mismo hilo verifica base de datos, MLflow (`/health`) y Airflow
  (`/api/v2/monitor/health`) con timeout de 3 s; `/health` responde el último estado (degradado
  en ≤ 20 s, recuperado en ≤ 20 s). Sin modelo cargado: 503.
- **Reentrenamiento:** la API crea el id, lo registra y dispara el DAG por la API REST de
  Airflow con ese id como `dag_run_id`. El DAG escribe el resultado en la base del producto;
  si una tarea falla, un callback lo marca `fallido`. Si Airflow no responde al disparar → 503.
  Duración observada ≈ 20–40 s (límite 10 min; `dagrun_timeout` = 10 min).
- **Arranque desde cero:** servicio `bootstrap` idempotente (esquema, carga del histórico
  marcada con su SHA-256, primer modelo solo si no existe el alias). Listo en ≈ 65 s.
- **Serialización:** `pickle` nativo y verificación de que el artefacto no dependa de `dill`
  (en Airflow, `dill` se registra en el pickler de Python que usa joblib y producía modelos
  que la API no podía cargar; se detectó en la simulación y la API siguió sirviendo el
  modelo anterior con `/health` degradado).
- **Versiones fijas:** imágenes `python:3.12.12-slim-bookworm`, `apache/airflow:3.3.2`,
  `ghcr.io/mlflow/mlflow:v3.17.2`, `postgres:16.10`; dependencias fijadas, scikit-learn
  1.9.1 idéntico en API y Airflow.
- **Credenciales** solo en `.env` (se entrega `.env.example`); los datos se montan desde
  `./data/publicados` en solo lectura.
