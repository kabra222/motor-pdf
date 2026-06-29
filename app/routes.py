from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.engine.background import (
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

router = APIRouter(
    tags=["Extração"],
)
cache = PDFCache()

MAX_FILE_SIZE = int(os.getenv("MOTOR_PDF_MAX_FILE_SIZE", str(50 * 1024 * 1024)))
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


async def _validate_file(file: UploadFile) -> tuple[bytes, str]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Formato inválido — envie um arquivo PDF")
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"Arquivo muito grande (máx {MAX_FILE_SIZE >> 20}MB)")
    return data, file.filename


# ── endpoints ────────────────────────────────────────────────────


@router.post(
    "/extract",
    response_model=ExtractionResult | ErrorResponse,
    summary="Extrair texto de PDF",
    description=(
        "Extrai texto, tabelas, imagens e metadados de um PDF. "
        "Suporta OCR (PaddleOCR ou EasyOCR), chunking semântico, "
        "BCPD segmenter, descrição de imagens via LLM vision, "
        "e classificação de blocos (heading/paragraph/noise)."
    ),
)
async def extract(
    file: UploadFile = File(..., description="Arquivo PDF para processar"),
    chunk_size: int = Form(default=2000, ge=100, le=32000, description="Tamanho alvo de cada chunk em caracteres"),
    chunk_overlap: int = Form(default=200, ge=0, le=4000, description="Sobreposição entre chunks"),
    use_ocr: str = Form("", description='OCR: "easyocr", "paddleocr" ou vazio para desligado'),
    password: str | None = Form(None, description="Senha para PDF protegido"),
    format: FormatType = Form(FormatType.text, description="Formato de saída (text/markdown/semantic_html)"),
    model: str = Form("gpt-4", description="Modelo para tokenização (tiktoken)"),
    extract_images: bool = Form(False, description="Extrair imagens do PDF"),
    use_bcpd: bool = Form(False, description="Usar BCPD (Breakpoint Detection) para segmentação"),
    describe_images: bool = Form(False, description="Descrever imagens via LLM vision (requer OPENROUTER_API_KEY)"),
    x_api_key: str = Depends(verify_api_key),
):
    _check_rate(x_api_key or "anonymous")
    data, filename = await _validate_file(file)

    ocr_val: bool | str = False
    if use_ocr and use_ocr.lower() not in ("", "false", "0"):
        ocr_val = use_ocr if use_ocr.lower() in ("easyocr", "paddleocr") else True

    cache_params = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_ocr": str(ocr_val),
        "model": model,
        "extract_images": extract_images,
        "use_bcpd": use_bcpd,
        "format": format.value if isinstance(format, FormatType) else str(format),
    }
    cache_key = cache.make_key(data, cache_params)
    cached = await cache.get(cache_key)
    if cached:
        try:
            extraction = ExtractionResult(**cached)
        except Exception:
            pass
        else:
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
                    classified_count=extraction.classified_count,
                    annotations=extraction.annotations,
                    links=extraction.links,
                )
            return extraction

    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = await asyncio.to_thread(
            extract_text,
            tmp_path,
            use_ocr=ocr_val,
            password=password,
            extract_images=extract_images,
        )
    except Exception as e:
        raise HTTPException(500, f"Erro na extração: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not result["num_pages"]:
        return ErrorResponse(error="PDF vazio")
    if result["num_pages"] > MAX_PAGES:
        return ErrorResponse(error="PDF excede limite de páginas")

    if not result["metadata"].get("title"):
        result["metadata"]["title"] = Path(filename).stem

    chunks = chunk_text(
        result["text"],
        pages_text=result.get("pages_text"),
        blocks=result.get("blocks"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
        use_bcpd=use_bcpd,
    )

    if describe_images and result.get("images") and os.getenv("OPENROUTER_API_KEY"):
        try:
            from app.engine.vision import describe_images_batch
            result["images"] = describe_images_batch(result["images"])
        except Exception:
            pass

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
        classified_count=result.get("classified_count", 0),
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
            classified_count=extraction.classified_count,
        )

    return extraction


@router.post(
    "/documents",
    status_code=201,
    summary="Enviar PDF para processamento assíncrono",
    description=(
        "Envia um PDF para processamento em background. "
        "Retorna um job_id imediatamente. Acompanhe o progresso via GET /documents/{id} "
        "ou via SSE em /documents/{id}/stream."
    ),
)
async def create_document(
    file: UploadFile = File(..., description="Arquivo PDF para processar"),
    chunk_size: int = Form(default=2000, ge=100, le=32000, description="Tamanho do chunk em caracteres"),
    chunk_overlap: int = Form(default=200, ge=0, le=4000, description="Sobreposição entre chunks"),
    use_ocr: str = Form("", description='OCR: "easyocr", "paddleocr" ou vazio'),
    password: str | None = Form(None, description="Senha para PDF protegido"),
    model: str = Form("gpt-4", description="Modelo para tokenização"),
    extract_images: bool = Form(False, description="Extrair imagens"),
    use_bcpd: bool = Form(False, description="Usar BCPD segmenter"),
    describe_images: bool = Form(False, description="Descrever imagens via LLM"),
    x_api_key: str = Depends(verify_api_key),
):
    _check_rate(x_api_key or "anonymous")
    data, filename = await _validate_file(file)
    job_id = await create_job()

    ocr_val: bool | str = False
    if use_ocr and use_ocr.lower() not in ("", "false", "0"):
        ocr_val = use_ocr if use_ocr.lower() in ("easyocr", "paddleocr") else True

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
            ocr_val,
            password,
            model,
            extract_images,
            use_bcpd,
            describe_images=describe_images,
        )
    )

    return {"id": job_id, "status": "processing"}


@router.get(
    "/documents/{job_id}",
    summary="Obter resultado do job",
    description="Retorna o resultado de um job de processamento assíncrono pelo seu ID.",
)
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


@router.get(
    "/documents/{job_id}/stream",
    summary="SSE: acompanhar progresso do job",
    description=(
        "Streaming Server-Sent Events com o progresso do job. "
        "Eventos: progress, complete, error. Timeout de 30s por evento."
    ),
)
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
            except TimeoutError:
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


@router.get(
    "/documents/{job_id}/chunks",
    summary="Listar chunks de um job com paginação",
    description=(
        "Retorna os chunks extraídos de um job já processado, "
        "com suporte a paginação via page e per_page."
    ),
)
async def get_document_chunks(
    job_id: str,
    page: int = Query(default=1, ge=1, description="Número da página"),
    per_page: int = Query(default=20, ge=1, le=100, description="Itens por página"),
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


@router.post(
    "/extract/quality",
    summary="Extrair apenas métricas de qualidade",
    description="Processa um PDF rapidamente e retorna apenas as métricas de qualidade da extração.",
)
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


@router.get(
    "/cache/stats",
    summary="Estatísticas do cache",
    description="Retorna o tamanho atual do cache de extrações.",
)
async def cache_stats():
    return {"size": await cache.size()}


@router.delete(
    "/cache",
    summary="Limpar cache",
    description="Remove todas as entradas do cache de extrações.",
)
async def clear_cache():
    await cache.clear()
    return {"status": "ok"}


@router.get(
    "/health",
    summary="Health check",
    description="Verifica o status do serviço, versão e configuração atual.",
)
async def health():
    from app.main import VERSION
    deploy_version = "unknown"
    try:
        deploy_version = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True, timeout=2
        ).stdout.strip() or "unknown"
    except Exception:
        deploy_version = "no-git"
    return {
        "status": "ok",
        "version": VERSION,
        "deploy": deploy_version,
        "max_file_size_mb": MAX_FILE_SIZE >> 20,
        "max_pages": MAX_PAGES,
        "rate_limit": f"{RATE_LIMIT}/{RATE_WINDOW}s",
        "api_key_required": bool(API_KEY),
    }
