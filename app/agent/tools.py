from __future__ import annotations

import math
import os
from typing import Optional

from app.agent.llm import LLMProvider
from app.agent.store import PersistentVectorStore

# ── Reranking ─────────────────────────────────────────────────────

_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        except Exception:
            _CROSS_ENCODER = False
    return _CROSS_ENCODER if _CROSS_ENCODER is not False else None


def rerank(
    query: str,
    results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if not results:
        return results
    ce = _get_cross_encoder()
    if ce is None:
        return results[:top_k]
    pairs = [(query, r["text"]) for r in results]
    scores = ce.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = round(float(s), 4)
    results.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    return results[:top_k]


# ── Query expansion ───────────────────────────────────────────────

EXPANSION_PROMPT = (
    "Gere {n} variações da pergunta abaixo para busca em um documento PDF. "
    "Cada variação deve ser uma reformulação ou desdobramento que ajude "
    "a encontrar informações relevantes no documento. "
    "Responda APENAS com as variações, uma por linha, sem numeração.\n\n"
    "Pergunta: {query}"
)


async def expand_query(
    query: str,
    llm: LLMProvider,
    n: int = 3,
) -> list[str]:
    try:
        result = await llm.chat([
            {
                "role": "system",
                "content": "Você gera variações de perguntas para busca em documentos.",
            },
            {
                "role": "user",
                "content": EXPANSION_PROMPT.format(n=n, query=query),
            },
        ])
        variants = [q.strip() for q in result.strip().split("\n") if q.strip()]
        return [query] + variants[:n]
    except Exception:
        return [query]


# ── Search with expansion ─────────────────────────────────────────

async def search(
    query: str,
    llm: LLMProvider,
    store: PersistentVectorStore,
    top_k: int = 5,
    use_expansion: bool = True,
) -> list[dict]:
    if use_expansion:
        queries = await expand_query(query, llm)
    else:
        queries = [query]

    all_results: list[dict] = []
    seen_ids: set[str] = set()
    for q in queries:
        query_emb = await llm.embed(q)
        results = store.search(query_emb, top_k=top_k * 2, threshold=0.0)
        for r in results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                all_results.append(r)

    all_results = rerank(query, all_results, top_k=top_k)
    return all_results


# ── Context building with metadata ────────────────────────────────

async def build_pdf_context(
    query: str,
    llm: LLMProvider,
    store: PersistentVectorStore,
    top_k: int = 5,
    use_expansion: bool = True,
    max_chars: int = 4000,
) -> tuple[str, list[dict]]:
    results = await search(query, llm, store, top_k=top_k, use_expansion=use_expansion)

    context_parts = []
    for r in results:
        meta = r.get("metadata", {})
        page = meta.get("page", "")
        section = meta.get("section") or ""
        heading = meta.get("heading") or ""
        loc = []
        if page is not None and page != "":
            loc.append(f"p.{page}")
        if heading:
            loc.append(f"§{heading[:50]}")
        loc_str = f" [{', '.join(loc)}]" if loc else ""
        score = r.get("rerank_score", r.get("score", 0))
        context_parts.append(
            f"[Trecho (score: {score}){loc_str}]\n{r['text']}\n"
        )
    context = "\n".join(context_parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[cortado]"
    return context, results


# ── Summarize ─────────────────────────────────────────────────────

async def summarize(
    text: str,
    llm: LLMProvider,
    max_length: int = 500,
    style: str = "paragraph",
) -> str:
    prompt = (
        f"Resuma o texto abaixo em {'um parágrafo' if style == 'paragraph' else 'tópicos'}. "
        f"Mantenha os números e dados principais. Máximo de {max_length} caracteres.\n\n{text}"
    )
    return await llm.chat([
        {"role": "system", "content": "Você é um assistente especializado em resumir documentos."},
        {"role": "user", "content": prompt},
    ])


# ── Classify ──────────────────────────────────────────────────────

async def classify(text: str, llm: LLMProvider) -> dict:
    prompt = (
        "Classifique o documento abaixo em uma das categorias: "
        "relatorio_financeiro, contrato, artigo_cientifico, "
        "fatura, manual_tecnico, apresentacao, currículo, carta, "
        "outro. Responda APENAS com um JSON: "
        '{"category": "...", "confidence": 0.0-1.0, "reason": "..."}\n\n'
        f"{text[:3000]}"
    )
    result = await llm.chat([
        {"role": "system", "content": "Você classifica documentos profissionais."},
        {"role": "user", "content": prompt},
    ])
    import json, re
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"category": "outro", "confidence": 0.0, "reason": "Falha ao classificar"}


# ── Extract entities ──────────────────────────────────────────────

async def extract_entities(
    text: str,
    llm: LLMProvider,
    schema: Optional[dict] = None,
) -> dict:
    import json
    if schema:
        schema_desc = json.dumps(schema, ensure_ascii=False)
        prompt = (
            f"Extraia do texto abaixo as entidades conforme este schema JSON:\n"
            f"{schema_desc}\n\nTexto:\n{text[:4000]}"
        )
    else:
        prompt = (
            "Extraia do texto abaixo: pessoas, empresas, valores monetários, "
            "datas, números de documento. Responda APENAS com um JSON.\n\n"
            f"{text[:4000]}"
        )

    result = await llm.chat([
        {"role": "system", "content": "Você extrai entidades estruturadas de documentos."},
        {"role": "user", "content": prompt},
    ])
    import json, re
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Falha ao extrair entidades", "raw": result}


# ── Tool definitions (OpenAI format) ──────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Busca trechos relevantes no documento PDF carregado. "
                "Use esta ferramenta quando precisar de informações específicas "
                "do documento para responder à pergunta do usuário."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A pergunta ou termo de busca",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": (
                "Gera um resumo do documento carregado. "
                "Use quando o usuário pedir um resumo geral."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_length": {
                        "type": "integer",
                        "description": "Número máximo de caracteres (padrão 500)",
                    },
                    "style": {
                        "type": "string",
                        "enum": ["paragraph", "bullets"],
                        "description": "Estilo do resumo",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_document",
            "description": (
                "Classifica o tipo do documento carregado "
                "(relatorio_financeiro, contrato, artigo_cientifico, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
