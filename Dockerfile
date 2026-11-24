FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY scripts/ ./scripts/

COPY alembic.ini ./
COPY alembic/ ./alembic/

RUN pip install --no-cache-dir .

RUN useradd -m appuser \
    && mkdir -p /app/model_cache /app/uploads \
    && chown appuser:appuser /app/model_cache /app/uploads

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
