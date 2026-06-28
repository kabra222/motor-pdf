from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agent.agent import PDFAgent
from app.agent.llm import create_provider
from app.agent.store import VectorStore
from app.engine.chunker import chunk_text
from app.engine.extractor import extract_text

agent_router = APIRouter()

DEFAULT_PROVIDER = os.getenv("MOTOR_PDF_LLM_PROVIDER", "openai")
_agent: PDFAgent | None = None
_agent_lock = asyncio.Lock()


async def get_agent() -> PDFAgent:
    global _agent
    if _agent is None:
        async with _agent_lock:
            if _agent is None:
                llm = create_provider(DEFAULT_PROVIDER)
                store = VectorStore()
                _agent = PDFAgent(llm=llm, store=store)
    return _agent


@agent_router.post("/agent/load")
async def agent_load(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=2000),
    chunk_overlap: int = Form(default=200),
    use_bcpd: bool = Form(False),
    use_ocr: str = Form(""),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Envie um arquivo PDF")

    data = await file.read()
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    ocr_val: bool | str = False
    if use_ocr and use_ocr.lower() not in ("", "false", "0"):
        ocr_val = use_ocr if use_ocr.lower() in ("easyocr", "paddleocr") else True

    try:
        result = extract_text(tmp_path, use_ocr=ocr_val)
        result["metadata"]["title"] = Path(file.filename).stem
        text = result["text"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    chunks = chunk_text(
        text,
        pages_text=result.get("pages_text"),
        blocks=result.get("blocks"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_bcpd=use_bcpd,
    )

    agent = await get_agent()
    agent.store.clear()

    embed_tasks = []
    for c in chunks:
        embed_tasks.append(agent.llm.embed(c.text))

    embeddings = await asyncio.gather(*embed_tasks)

    metadata = [
        {"page": c.page, "section": c.section, "heading": c.heading, "chunk_index": c.index}
        for c in chunks
    ]
    agent.index_chunks([c.text for c in chunks], embeddings, metadata)

    return {
        "status": "loaded",
        "title": result["metadata"]["title"],
        "chunks": len(chunks),
        "pages": result["num_pages"],
        "quality": result.get("quality"),
    }


@agent_router.post("/agent/chat")
async def agent_chat(
    message: str = Form(...),
    stream: bool = Form(False),
    history_json: str = Form("[]"),
):
    agent = await get_agent()
    history = json.loads(history_json) if history_json else []

    response = await agent.chat(message, history=history, stream=stream)

    if stream:
        async def event_stream():
            full = ""
            async for token in response:
                full += token
                yield f"data: {json.dumps({'token': token, 'full': full})}\n\n"
            yield f"data: {json.dumps({'token': '', 'full': full, 'done': True})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return {"response": response}


@agent_router.post("/agent/summarize")
async def agent_summarize(
    max_length: int = Form(500),
    style: str = Form("paragraph"),
):
    agent = await get_agent()
    if agent.store.size == 0:
        raise HTTPException(400, "Nenhum documento carregado. Use /agent/load primeiro.")

    store_text = "\n".join(
        e.text for e in agent.store._entries
    )
    summary = await agent.summarize(store_text, max_length=max_length, style=style)
    return {"summary": summary}


@agent_router.post("/agent/classify")
async def agent_classify():
    agent = await get_agent()
    if agent.store.size == 0:
        raise HTTPException(400, "Nenhum documento carregado.")

    store_text = "\n".join(
        e.text for e in agent.store._entries
    )[:4000]
    result = await agent.classify(store_text)
    return result


@agent_router.post("/agent/extract")
async def agent_extract(schema_json: str = Form("{}")):
    agent = await get_agent()
    if agent.store.size == 0:
        raise HTTPException(400, "Nenhum documento carregado.")

    store_text = "\n".join(
        e.text for e in agent.store._entries
    )[:4000]
    schema = json.loads(schema_json) if schema_json else None
    result = await agent.extract(store_text, schema=schema)
    return result


@agent_router.get("/agent/status")
async def agent_status():
    agent = await get_agent()
    return {
        "provider": DEFAULT_PROVIDER,
        "indexed_chunks": agent.store.size,
        "model": getattr(agent.llm, "model", "unknown"),
    }
