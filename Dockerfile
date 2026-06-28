FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir PyMuPDF pdfplumber fastapi uvicorn pydantic python-multipart tiktoken pypdf

COPY pyproject.toml .
COPY app/ app/

RUN pip install --no-cache-dir ".[openai,anthropic,advanced]"

COPY . .

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
