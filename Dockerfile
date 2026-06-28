FROM python:3.12-slim

WORKDIR /app

COPY app/ app/
COPY pyproject.toml .

RUN pip install --no-cache-dir PyMuPDF pdfplumber fastapi uvicorn pydantic python-multipart tiktoken pypdf ".[openai,anthropic,advanced,tables,easyocr]"

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
