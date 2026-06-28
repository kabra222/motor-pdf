from __future__ import annotations

import math
from typing import Optional

from app.agent.llm import LLMProvider
from app.agent.store import VectorStore


async def search(
    query: str,
    llm: LLMProvider,
    store: VectorStore,
    top_k: int = 5,
) -> list[dict]:
    query_emb = await llm.embed(query)
    results = store.search(query_emb, top_k=top_k)
    return results


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


async def extract_entities(text: str, llm: LLMProvider, schema: Optional[dict] = None) -> dict:
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


async def build_pdf_context(
    query: str,
    llm: LLMProvider,
    store: VectorStore,
    max_chars: int = 4000,
) -> tuple[str, list[dict]]:
    results = await search(query, llm, store)
    context_parts = []
    for r in results:
        context_parts.append(
            f"[Trecho (score: {r['score']})]\n{r['text']}\n"
        )
    context = "\n".join(context_parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[cortado]"
    return context, results
