# Contexto para agentes de programación

Servicio de pronóstico diario de ventas (Laboratorio 2, Product Development). El contrato
HTTP del enunciado (Anexo) es inviolable: se pueden agregar campos a las respuestas, nunca
omitir ni renombrar los del contrato.

## Estructura
- `src/novamercado/`: paquete único. Los endpoints (`api.py`) solo orquestan; la limpieza
  vive en `limpieza.py` (Python puro, sin FastAPI ni BD) y se valida luego con Pydantic
  (`esquemas.py`).
- `features.py` es la ÚNICA fuente de variables: se usa al entrenar y al predecir. Las
  variables de la fecha F solo pueden usar ventas de fechas < F.
- `entrenamiento.py` contiene el reentrenamiento y el criterio de reemplazo; lo llaman el
  DAG (`dags/`) y `bootstrap.py`. El alias `campeon` se mueve solo al final.
- `registro.py` serializa con `pickle` nativo (no joblib): en Airflow `dill` contamina el
  pickler de Python y la API no podría cargar el modelo.

## Reglas
- Datos (`data/`) y credenciales (`.env`) nunca al repositorio ni a las imágenes.
- Imágenes y dependencias con versiones fijas; scikit-learn igual en API y Airflow.
- No usar el reloj del sistema para validar fechas: el rango válido sale de `feriados.csv`.
- Toda política nueva ante datos defectuosos: regla general en `limpieza.py`, prueba en
  `tests/test_limpieza.py` y fila en la tabla de `DECISIONES.md`.
- Las pruebas no deben depender de MLflow, Airflow, Postgres ni de los datos publicados.

## Comandos
- Pruebas: `docker compose run --rm --no-deps api python -m pytest -q`
- Levantar: `docker compose up -d --build` · Salud: `curl localhost:8000/health`
- E2E: `python3 scripts/prueba_integral.py` · Simulación: `python3 scripts/simular_operacion.py data/publicados`
