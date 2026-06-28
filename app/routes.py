from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.engine.background import (
    cleanup_job,
    create_job,
    get_job,
    get_job_events,
    start_processing,
)
from app.engine.cache import PDFCache
from app.engine.chunker import chunk_text
from app.engine.extractor import extract_text
from app.engine.formatter import format_result
from app.models import (
    ErrorResponse,
    ExtractionResult,
    FormatType,
    JobResult,
    JobStatus,
)

router = APIRouter()
cache = PDFCache()

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_PAGES = 500
API_KEY = os.getenv("MOTOR_PDF_API_KEY", "")

# ── rate limiter ──────────────────────────────────────────────────
_rates: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = int(os.getenv("MOTOR_PDF_RATE_LIMIT", "60"))
RATE_WINDOW = int(os.getenv("MOTOR_PDF_RATE_WINDOW", "60"))


def _check_rate(key: str) -> None:
    now = time.time()
    window_ago = now - RATE_WINDOW
    _rates[key] = [t for t in _rates[key] if t > window_ago]
    if len(_rates[key]) >= RATE_LIMIT:
        raise HTTPException(429, "Muitas requisições. Aguarde e tente novamente.")
    _rates[key].append(now)


async def verify_api_key(x_api_key: str = Header("")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(403, "API key inválida")
    return x_api_key


# ── file validation ──────────────────────────────────────────────


async def _validate_file(file: UploadFile) -> tuple[bytes, str]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Formato inválido — envie um arquivo PDF")
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"Arquivo muito grande (máx {MAX_FILE_SIZE >> 20}MB)")
    return data, file.filename


# ── endpoints ────────────────────────────────────────────────────


@router.post("/extract", response_model=ExtractionResult | ErrorResponse)
async def extract(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=2000, ge=100, le=32000),
    chunk_overlap: int = Form(default=200, ge=0, le=4000),
    use_ocr: bool = Form(False),
    password: str | None = Form(None),
    format: FormatType = Form(FormatType.text),
    model: str = Form("gpt-4"),
    extract_images: bool = Form(False),
    x_api_key: str = Depends(verify_api_key),
):
    _check_rate(x_api_key or "anonymous")
    data, filename = await _validate_file(file)

    cache_params = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_ocr": use_ocr,
        "model": model,
        "extract_images": extract_images,
    }
    cache_key = cache.make_key(data, cache_params)
    cached = await cache.get(cache_key)
    if cached:
        return ExtractionResult(**cached)

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = extract_text(
            tmp_path,
            use_ocr=use_ocr,
            password=password,
            extract_images=extract_images,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not result["num_pages"]:
        return ErrorResponse(error="PDF vazio")
    if result["num_pages"] > MAX_PAGES:
        return ErrorResponse(error="PDF excede limite de páginas")

    result["metadata"]["title"] = Path(filename).stem

    chunks = chunk_text(
        result["text"],
        pages_text=result.get("pages_text"),
        blocks=result.get("blocks"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
    )

    extraction = ExtractionResult(
        filename=filename,
        text=result["text"],
        chunks=chunks,
        blocks=result.get("blocks", []),
        headings=result.get("headings", []),
        tables=result.get("tables", []),
        images=result.get("images", []),
        metadata=result["metadata"],
        num_pages=result["num_pages"],
        num_chunks=len(chunks),
        scanned_pages=result.get("scanned_pages", []),
        quality=result.get("quality"),
    )

    await cache.set(cache_key, extraction.model_dump())

    if format != FormatType.text:
        formatted = format_result(
            extraction.text,
            extraction.blocks,
            extraction.headings,
            extraction.tables,
            format,
        )
        return ExtractionResult(
            filename=extraction.filename,
            text=formatted,
            chunks=extraction.chunks,
            blocks=extraction.blocks,
            headings=extraction.headings,
            tables=extraction.tables,
            images=extraction.images,
            metadata=extraction.metadata,
            num_pages=extraction.num_pages,
            num_chunks=extraction.num_chunks,
            scanned_pages=extraction.scanned_pages,
            quality=extraction.quality,
        )

    return extraction


@router.post("/documents", status_code=201)
async def create_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=2000, ge=100, le=32000),
    chunk_overlap: int = Form(default=200, ge=0, le=4000),
    use_ocr: bool = Form(False),
    password: str | None = Form(None),
    model: str = Form("gpt-4"),
    extract_images: bool = Form(False),
    x_api_key: str = Depends(verify_api_key),
):
    _check_rate(x_api_key or "anonymous")
    data, filename = await _validate_file(file)
    job_id = await create_job()

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    asyncio.create_task(
        start_processing(
            job_id,
            tmp_path,
            filename,
            chunk_size,
            chunk_overlap,
            use_ocr,
            password,
            model,
            extract_images,
        )
    )

    return {"id": job_id, "status": "processing"}


@router.get("/documents/{job_id}")
async def get_document(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    return JobResult(
        id=job.id,
        status=job.status,
        result=job.result,
        error=job.error,
        progress=job.progress,
        total=job.total,
    )


@router.get("/documents/{job_id}/stream")
async def stream_document(job_id: str):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")

    queue = await get_job_events(job_id)
    if queue is None:
        raise HTTPException(404, "Job não encontrado")

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                if event["event"] in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield f"event: keepalive\ndata: {json.dumps({'ts': __import__('time').time()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/documents/{job_id}/chunks")
async def get_document_chunks(
    job_id: str,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado")
    if job.status != JobStatus.done or not job.result:
        raise HTTPException(400, "Processamento ainda em andamento")

    chunks = job.result.chunks
    total = len(chunks)
    start = (page - 1) * per_page
    end = start + per_page

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "chunks": chunks[start:end],
    }


@router.post("/extract/quality")
async def extract_quality(
    file: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key),
):
    _check_rate(x_api_key or "anonymous")
    data, filename = await _validate_file(file)

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = extract_text(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result.get("quality", {})


@router.get("/cache/stats")
async def cache_stats():
    return {"size": await cache.size()}


@router.delete("/cache")
async def clear_cache():
    await cache.clear()
    return {"status": "ok"}


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.4.0",
        "max_file_size_mb": MAX_FILE_SIZE >> 20,
        "max_pages": MAX_PAGES,
        "rate_limit": f"{RATE_LIMIT}/{RATE_WINDOW}s",
        "api_key_required": bool(API_KEY),
    }
