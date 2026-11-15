# NovaMercado — Servicio de pronóstico diario de ventas

Servicio de ML en producción que pronostica la venta diaria por familia de producto de
cada tienda, defiende su contrato ante datos defectuosos, registra lo que predice y lo que
realmente ocurre, y se reentrena con esa información. Integra **FastAPI**, **PostgreSQL**,
**MLflow** y **Apache Airflow**, todo levantado con `docker compose up`.

## Versión entregada

| | |
|---|---|
| Laboratorio | Laboratorio 2 — Servicio de ML en Producción (Product Development) |
| Tag de entrega | **`entrega-lab2`** |
| Versión del producto | **1.0.0** (`pyproject.toml`, imágenes `novamercado/api:1.0.0` y `novamercado/airflow:3.3.2`) |
| Release | [entrega-lab2](https://github.com/carmenmena5991-ctrl/novamercado/releases/tag/entrega-lab2) |

La versión que se evalúa es exactamente el commit al que apunta el tag `entrega-lab2`:

```bash
git clone https://github.com/carmenmena5991-ctrl/novamercado.git && cd novamercado
git checkout entrega-lab2
git log -1 --format='%h %ci %s'   # commit entregado
```

## Requisitos

- Docker Engine / Docker Desktop con Compose v2.
- ~5 GB de RAM libres para los contenedores.
- Puertos libres: `8000` (API), `5000` (MLflow), `8080` (Airflow), `5433` (base del producto).

## Levantar desde cero

```bash
git clone https://github.com/carmenmena5991-ctrl/novamercado.git && cd novamercado
git checkout entrega-lab2

# 1. Credenciales locales (no se versionan)
cp .env.example .env            # cambiar las claves si se desea

# 2. Datos publicados: descomprimir el archivo del GES en ./data/publicados/
#    Debe quedar así:
#      data/publicados/tiendas.csv
#      data/publicados/familias.csv
#      data/publicados/historico_ventas.csv
#      data/publicados/feriados.csv
#      data/publicados/costos.csv
#      data/publicados/catalogo_errores.md   (opcional)
#      data/publicados/ejemplos/             (opcional)
mkdir -p data/publicados && unzip datos_lab2.zip -d data/publicados

# 3. Construir y levantar
docker compose up -d --build

# 4. Esperar a que esté listo (≈1-2 min después de construir las imágenes)
curl localhost:8000/health        # {"estado":"ok","version_modelo":"1"}
```

En el primer arranque el servicio `bootstrap` crea el esquema, carga el histórico en la base
del producto y entrena y registra el primer modelo (alias `campeon`). Al volver a levantar
(`docker compose down && docker compose up -d`, sin `-v`) detecta lo existente y no repite
nada; la API carga el modelo desde su caché local aunque MLflow aún no haya arrancado.

| Servicio | URL | Notas |
|---|---|---|
| API | http://localhost:8000/docs | contrato del Anexo |
| MLflow | http://localhost:5000 | experimento `novamercado_pronostico`, modelo `pronostico_ventas@campeon` |
| Airflow | http://localhost:8080 | usuario `admin`, clave `AIRFLOW_ADMIN_PASSWORD` de `.env`; DAG `reentrenamiento_pronostico` |

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/v1/prediccion` | pronóstico de un día para una tienda (+ `calidad`) |
| POST | `/v1/observaciones` | ventas reales de un día (idempotente) |
| POST | `/admin/reentrenar` | dispara el DAG; 202 con `id`, 503 si no se puede iniciar |
| GET | `/admin/reentrenar/{id}` | `en_curso` / `exitoso` / `fallido` + `promovido`, `version`, `motivo` |
| GET | `/modelo` | `nombre`, `alias`, `version` del modelo en uso |
| GET | `/health` | `ok` o `degradado` + `motivo`; 503 sin modelo |

```bash
curl -X POST localhost:8000/v1/prediccion -H 'Content-Type: application/json' \
     -d @data/publicados/ejemplos/prediccion_valida.json
```

## Pruebas

Las pruebas no requieren MLflow, Airflow, Postgres ni los datos publicados (usan catálogos
sintéticos y dobles en memoria):

```bash
docker compose run --rm --no-deps api python -m pytest -q
# o, con un entorno local:  pip install -r docker/api/requirements.txt && pytest -q
```

Pruebas de punta a punta contra el producto levantado (solo biblioteca estándar):

```bash
python3 scripts/prueba_integral.py data/publicados/ejemplos evidencia/respuestas
python3 scripts/simular_operacion.py data/publicados 21 2026-03-01 8
```

## Estructura

```
src/novamercado/      código del producto
  limpieza.py         defensa del contrato (reparar / advertir / rechazar)
  esquemas.py         modelos Pydantic del contrato
  features.py         variables del modelo (las mismas al entrenar y al predecir)
  modelo.py           baseline, gradient boosting, cantidad sugerida y métricas de costo
  entrenamiento.py    reentrenamiento + criterio de reemplazo (lo usan el DAG y bootstrap)
  registro.py         MLflow Model Registry y caché local del modelo
  servicio.py         recarga en caliente del modelo y salud de dependencias
  api.py / main.py    FastAPI
  db.py               PostgreSQL del producto
  bootstrap.py        arranque desde cero idempotente
dags/                 DAG de reentrenamiento (Airflow 3)
docker/               Dockerfiles y dependencias con versiones fijas
tests/                pytest
scripts/              pruebas de punta a punta y simulación de operación
evidencia/            EVIDENCIA.md (resumen), respuestas de la API (válidas y defectuosas),
                      versiones y runs de MLflow y ejecuciones del DAG exportadas en JSON,
                      y la simulación de 21 días
DECISIONES.md         decisiones técnicas y de producto
```

## Operación

- **MLflow caído:** la API sigue respondiendo con el modelo cargado; `/health` pasa a
  `degradado` en ≤ 20 s y vuelve a `ok` en ≤ 20 s cuando MLflow regresa.
- **Promoción:** la API consulta el alias cada 15 s y cambia de modelo sin reiniciarse.
- **Reiniciar sin perder datos:** `docker compose down && docker compose up -d`
  (los volúmenes `nova_db`, `mlflow_*`, `airflow_*` y `cache_modelo` se conservan).
- **Borrar todo:** `docker compose down -v`.
