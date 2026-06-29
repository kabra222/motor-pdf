from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.agent.llm import LLMProvider
from app.agent.tools import (
    TOOLS,
    build_pdf_context,
    classify,
    deep_analyze,
    extract_entities,
    extract_structure,
    search,
    summarize,
)

from .store import (
    PersistentVectorStore,
    get_history,
)


class PDFAgent:
    def __init__(
        self,
        llm: LLMProvider,
        store: PersistentVectorStore | None = None,
    ):
        self.llm = llm
        self.store = store or PersistentVectorStore()
        self.extraction_result: dict | None = None

    async def chat(
        self,
        message: str,
        history: list[dict] | None = None,
        pdf_text: str | None = None,
        stream: bool = False,
        session_id: str | None = None,
        use_tools: bool = True,
    ) -> str | AsyncIterator[str]:
        if session_id:
            db_history = get_history(session_id, limit=20)
            history = (history or []) + db_history
        history = history or []

        if self.store.size > 0:
            context, results = await build_pdf_context(
                message, self.llm, self.store, use_expansion=True
            )
            system_prompt = (
                "Você é um analista de documentos PDF de alto nível. "
                "Use o contexto fornecido para responder perguntas sobre o documento. "
                "Sempre cite os trechos relevantes que usou para responder, "
                "incluindo o número da página e seção quando disponível. "
                "Se não souber a resposta, diga que não encontrou a informação no documento.\n\n"
                "INSTRUÇÕES DE ANÁLISE:\n"
                "- Para perguntas gerais sobre o documento: use analyze_document para análise profunda\n"
                "- Para pedidos de estrutura/organização: use extract_structure\n"
                "- Para resumos: use summarize_document\n"
                "- Para informações específicas: use search_document\n"
                "- Para classificação: use classify_document\n\n"
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

        if use_tools and self.store.size > 0:
            resp, tool_results = await self._tool_call_loop(messages)
        else:
            resp = await self.llm.chat(messages, stream=stream)

        return resp

    async def _tool_call_loop(
        self,
        messages: list[dict],
        max_turns: int = 3,
    ) -> tuple[str, list[dict]]:
        tools_desc = (
            "\n\nVocê tem acesso às seguintes ferramentas. Quando precisar de "
            "informações do documento, use UMA das opções abaixo. "
            "Responda SEMPRE em português.\n\n"
        )
        for t in TOOLS:
            fn = t["function"]
            tools_desc += f"- {fn['name']}: {fn['description']}\n"
            params = fn.get("parameters", {}).get("properties", {})
            if params:
                tools_desc += "  Parâmetros:\n"
                for pname, pinfo in params.items():
                    tools_desc += f"    {pname}: {pinfo.get('description', '')}\n"
            tools_desc += "\n"

        tools_desc += (
            "Para usar uma ferramenta, responda EXATAMENTE neste formato JSON "
            "(sem markdown, sem texto extra):\n"
            '{"tool": "nome_da_ferramenta", "arguments": {"param": "valor"}}\n\n'
            "Se não precisar de ferramenta, responda normalmente."
        )

        messages[-1]["content"] = messages[-1]["content"] + tools_desc

        turn = 0
        tool_results: list[dict] = []
        while turn < max_turns:
            turn += 1
            response_msg = await self.llm.chat(messages, stream=False)
            content = response_msg.strip() if isinstance(response_msg, str) else ""

            try:
                parsed = json.loads(content)
                tool_name = parsed.get("tool")
                arguments = parsed.get("arguments", {})
            except (json.JSONDecodeError, TypeError):
                return content, tool_results

            if not tool_name:
                return content, tool_results

            result = await self._dispatch_tool(tool_name, arguments)
            tool_results.append({"tool": tool_name, "result": result})

            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"Resultado da ferramenta '{tool_name}': "
                    f"{json.dumps(result, ensure_ascii=False)[:2000]}\n\n"
                    "Agora responda ao usuário com base neste resultado."
                ),
            })

        final = await self.llm.chat(messages)
        return final if isinstance(final, str) else "", tool_results

    async def _dispatch_tool(self, name: str, args: dict) -> dict:
        if name == "search_document":
            query = args.get("query", "")
            results = await search(query, self.llm, self.store, use_expansion=False)
            return {"results": results}
        elif name == "summarize_document":
            text = self._store_text_fallback()
            if not text:
                return {"error": "Nenhum texto disponível"}
            structure = self.extraction_result if self.extraction_result else None
            result = await summarize(
                text,
                self.llm,
                max_length=args.get("max_length", 500),
                style=args.get("style", "paragraph"),
                structure=structure,
            )
            return {"summary": result}
        elif name == "classify_document":
            text = self._store_text_fallback()[:4000]
            if not text:
                return {"error": "Nenhum texto disponível"}
            return await classify(text, self.llm)
        elif name == "analyze_document":
            if not self.extraction_result:
                return {"error": "Nenhum documento carregado. Use /agent/load primeiro."}
            return await deep_analyze(self.extraction_result, self.llm)
        elif name == "extract_structure":
            if not self.extraction_result:
                return {"error": "Nenhum documento carregado. Use /agent/load primeiro."}
            return await extract_structure(self.extraction_result, self.llm)
        return {"error": f"Ferramenta desconhecida: {name}"}

    def _store_text_fallback(self) -> str:
        try:
            entries = self.store.search(
                [0.0] * 384,
                top_k=100, threshold=-1.0,
            )
            return "\n".join(e.get("text", "") for e in entries)
        except Exception:
            return ""

    async def summarize(
        self,
        text: str,
        max_length: int = 500,
        style: str = "paragraph",
    ) -> str:
        return await summarize(text, self.llm, max_length, style)

    async def classify(self, text: str) -> dict:
        return await classify(text, self.llm)

    async def extract(self, text: str, schema: dict | None = None) -> dict:
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

    def index_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: list[dict] | None = None,
    ) -> None:
        for i, (text, emb) in enumerate(zip(chunks, embeddings, strict=False)):
            meta = metadata[i] if metadata else {"chunk_index": i}
            self.store.add(f"chunk_{i}", text, emb, meta)

    async def close(self):
        if hasattr(self.llm, "close"):
            await self.llm.close()
