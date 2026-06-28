FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY app/ app/
RUN pip install --no-cache-dir ".[tables,classifier,openai,anthropic,advanced]"

COPY . .

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
