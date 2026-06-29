from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.routes import agent_router
from app.engine.storage import init_db
from app.routes import router

VERSION = "0.6.0"

try:
    import sentry_sdk
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        traces_sample_rate=0.1,
        environment=os.getenv("RAILWAY_ENVIRONMENT", "production"),
    )
except Exception:
    pass

app = FastAPI(
    title="Motor PDF",
    description=(
        "Motor de extração de texto de PDFs para consumo por LLMs. "
        "Extrai texto, tabelas, imagens e metadados de PDFs com suporte "
        "a OCR, classificação semântica de blocos, análise de layout, "
        "e agente IA com RAG via OpenRouter."
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "Motor PDF", "url": "https://github.com/kabra222/motor-pdf"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    import traceback
    detail = str(exc)
    if not detail or detail == "Erro interno":
        detail = traceback.format_exc()[-500:]
    return JSONResponse(
        status_code=500,
        content={"error": "Erro interno", "detail": detail},
    )


app.include_router(router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
