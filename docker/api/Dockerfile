# Imagen de la API y del arranque inicial (bootstrap). Versiones fijas.
FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /cache_modelo \
    && chown app /cache_modelo

WORKDIR /app
COPY docker/api/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY src/ /app/src/
COPY tests/ /app/tests/
COPY pyproject.toml /app/

USER app
EXPOSE 8000
CMD ["uvicorn", "novamercado.main:app", "--host", "0.0.0.0", "--port", "8000"]
