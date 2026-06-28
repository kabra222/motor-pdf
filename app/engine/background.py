from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.engine.chunker import chunk_text
from app.engine.extractor import extract_text
from app.engine.storage import (
    create_job as storage_create_job,
    get_job as storage_get_job,
    update_job as storage_update_job,
    cleanup_stale_jobs as storage_cleanup,
)
from app.models import ExtractionResult, JobStatus

_job_events: dict[str, asyncio.Queue] = {}
_lock = asyncio.Lock()


async def create_job() -> str:
    job_id = await storage_create_job()
    async with _lock:
        _job_events[job_id] = asyncio.Queue()
    return job_id


async def get_job(job_id: str):
    return await storage_get_job(job_id)


async def get_job_events(job_id: str) -> asyncio.Queue | None:
    async with _lock:
        return _job_events.get(job_id)


async def _publish(job_id: str, event: str, data: dict) -> None:
    async with _lock:
        queue = _job_events.get(job_id)
    if queue:
        await queue.put({"event": event, "data": data})


async def start_processing(
    job_id: str,
    tmp_path: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
    use_ocr: bool,
    password: str | None,
    model: str,
    extract_images: bool,
) -> None:
    loop = asyncio.get_running_loop()

    def progress(page: int, total: int, stage: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _publish(job_id, "progress", {
                "page": page,
                "total": total,
                "stage": stage,
            }),
            loop,
        )
        asyncio.run_coroutine_threadsafe(
            storage_update_job(job_id, progress=page, total=total),
            loop,
        )

    try:
        result = await asyncio.to_thread(
            extract_text,
            tmp_path,
            use_ocr=use_ocr,
            password=password,
            extract_images=extract_images,
            progress=progress,
        )
        result["metadata"]["title"] = Path(filename).stem
    except Exception as e:
        await _publish(job_id, "error", {"error": str(e)})
        await storage_update_job(job_id, status=JobStatus.error, error=str(e))
        return

    try:
        chunks = chunk_text(
            result["text"],
            pages_text=result.get("pages_text"),
            blocks=result.get("blocks"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=model,
        )
    except Exception as e:
        await _publish(job_id, "error", {"error": f"Chunking failed: {e}"})
        await storage_update_job(
            job_id, status=JobStatus.error, error=f"Chunking failed: {e}"
        )
        return

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

    await storage_update_job(
        job_id,
        status=JobStatus.done,
        result=extraction,
        progress=result["num_pages"],
        total=result["num_pages"],
    )
    await _publish(job_id, "complete", json.loads(extraction.model_dump_json()))


async def cleanup_job(job_id: str) -> None:
    async with _lock:
        _job_events.pop(job_id, None)


async def cleanup_stale_jobs(max_age: int = 300) -> int:
    return await storage_cleanup(max_age)
