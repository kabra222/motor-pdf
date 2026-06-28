from __future__ import annotations

from typing import AsyncIterator, Optional

from app.agent.llm import LLMProvider
from app.agent.store import VectorStore
from app.agent.tools import (
    build_pdf_context,
    classify,
    extract_entities,
    search,
    summarize,
)


class PDFAgent:
    def __init__(
        self,
        llm: LLMProvider,
        store: Optional[VectorStore] = None,
    ):
        self.llm = llm
        self.store = store or VectorStore()

    async def chat(
        self,
        message: str,
        history: list[dict] | None = None,
        pdf_text: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        history = history or []

        if self.store.size > 0:
            context, results = await build_pdf_context(message, self.llm, self.store)
            system_prompt = (
                "Você é um assistente especializado em analisar documentos PDF. "
                "Use o contexto fornecido para responder perguntas sobre o documento. "
                "Sempre cite os trechos relevantes que usou para responder. "
                "Se não souber a resposta, diga que não encontrou a informação no documento.\n\n"
                f"Contexto do documento:\n{context}"
            )
        elif pdf_text:
            system_prompt = (
                "Você é um assistente especializado em analisar documentos PDF. "
                f"Use o texto fornecido para responder.\n\n{pdf_text[:8000]}"
            )
        else:
            system_prompt = (
                "Você é um assistente especializado em analisar documentos PDF. "
                "Nenhum documento foi carregado ainda."
            )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": message})

        return await self.llm.chat(messages, stream=stream)

    async def summarize(
        self,
        text: str,
        max_length: int = 500,
        style: str = "paragraph",
    ) -> str:
        return await summarize(text, self.llm, max_length, style)

    async def classify(self, text: str) -> dict:
        return await classify(text, self.llm)

    async def extract(self, text: str, schema: Optional[dict] = None) -> dict:
        return await extract_entities(text, self.llm, schema)

    async def query(self, text: str, query: str) -> str:
        if self.store.size > 0:
            context, _ = await build_pdf_context(query, self.llm, self.store)
        else:
            context = text[:4000]

        prompt = (
            f"Documento:\n{context}\n\n"
            f"Pergunta: {query}\n\n"
            "Responda com base APENAS no documento fornecido."
        )
        return await self.llm.chat([
            {"role": "system", "content": "Você analisa documentos PDF."},
            {"role": "user", "content": prompt},
        ])

    def index_chunks(self, chunks: list[str], embeddings: list[list[float]], metadata: list[dict] | None = None) -> None:
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            meta = metadata[i] if metadata else {"chunk_index": i}
            self.store.add(f"chunk_{i}", text, emb, meta)

    async def close(self):
        if hasattr(self.llm, "close"):
            await self.llm.close()
