from __future__ import annotations

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
    for r, s in zip(results, scores, strict=False):
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
        meta.get("section") or ""
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
    structure: dict | None = None,
) -> str:
    if structure and structure.get("headings"):
        toc = "\n".join(
            f"{'  ' * (h.get('level', 1) - 1)}- {h.get('text', '').strip()[:80]}"
            for h in structure["headings"][:30]
        )
        prompt = (
            "Você é um analista de documentos. Gere um resumo DETALHADO e "
            "estruturado do documento abaixo.\n\n"
            f"## Estrutura do documento:\n{toc}\n\n"
            f"## Texto completo:\n{text[:12000]}\n\n"
            "---\n\n"
            "Gere um resumo que inclua:\n"
            "1. **Propósito e escopo** do documento\n"
            "2. **Pontos principais** por seção\n"
            "3. **Conclusões ou achados** relevantes\n"
            "4. **Dados e números** importantes\n\n"
            f"Máximo de {max_length} caracteres. "
            f"Estilo: {'tópicos' if style == 'bullets' else 'parágrafos'}."
        )
    else:
        prompt = (
            f"Resuma o texto abaixo em {'um parágrafo' if style == 'paragraph' else 'tópicos'}. "
            f"Mantenha os números e dados principais. Máximo de {max_length} caracteres.\n\n{text[:8000]}"
        )
    return await llm.chat([
        {"role": "system", "content": "Você é um analista de documentos sênior."},
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
    import json
    import re
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"category": "outro", "confidence": 0.0, "reason": "Falha ao classificar"}


# ── Extract entities ──────────────────────────────────────────────

async def extract_entities(
    text: str,
    llm: LLMProvider,
    schema: dict | None = None,
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
    import json
    import re
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Falha ao extrair entidades", "raw": result}


# ── Deep analysis ─────────────────────────────────────────────────

async def deep_analyze(
    extraction_result: dict,
    llm: LLMProvider,
) -> dict:
    headings = extraction_result.get("headings", [])
    blocks = extraction_result.get("blocks", [])
    tables = extraction_result.get("tables", [])
    quality = extraction_result.get("quality", {})
    metadata = extraction_result.get("metadata", {})
    text = extraction_result.get("text", "")
    num_pages = extraction_result.get("num_pages", 0)

    toc = []
    for h in headings:
        ht = h.get("text", "").strip().split("\n")[0][:80]
        toc.append({
            "level": h.get("level", 0),
            "text": ht,
            "page": h.get("page", 0),
        })

    sections: list[dict] = []
    current_section: dict | None = None
    for b in blocks:
        if b.get("type") != "text":
            continue
        is_heading = b.get("is_heading") or b.get("layout_type") == "heading"
        if is_heading:
            if current_section and current_section.get("content"):
                sections.append(current_section)
            ht = b.get("text", "").strip().split("\n")[0][:100]
            current_section = {
                "heading": ht,
                "level": b.get("heading_level", 4),
                "page": b.get("page", 0),
                "content": "",
                "blocks": 0,
            }
        elif current_section is not None:
            current_section["content"] += b.get("text", "") + "\n"
            current_section["blocks"] += 1
        else:
            current_section = {
                "heading": "(Pré-texto)",
                "level": 0,
                "page": b.get("page", 0),
                "content": b.get("text", "") + "\n",
                "blocks": 1,
            }
    if current_section:
        sections.append(current_section)

    classified = {}
    for b in blocks:
        if b.get("type") == "text":
            lt = b.get("layout_type", "unknown")
            classified[lt] = classified.get(lt, 0) + 1

    text_blocks = [b for b in blocks if b.get("type") == "text"]

    section_analysis_parts = []
    char_budget = 8000
    for sec in sections[:20]:
        content = sec["content"].strip()
        if len(content) < 30:
            continue
        chunk = f"### {sec['heading']} (p.{sec['page']})\n{content[:600]}\n"
        if len("\n".join(section_analysis_parts)) + len(chunk) > char_budget:
            break
        section_analysis_parts.append(chunk)

    section_context = "\n".join(section_analysis_parts)

    toc_text = ""
    for t in toc:
        indent = "  " * (t["level"] - 1) if t["level"] > 0 else ""
        toc_text += f"{indent}- {t['text']} (p.{t['page']})\n"

    analysis_prompt = (
        "Analise o documento abaixo e forneça uma análise COMPLETA em português.\n\n"
        f"Documento: {num_pages} páginas, {len(text_blocks)} blocos, "
        f"{len(headings)} seções, {len(tables)} tabelas\n"
        f"Qualidade: {quality.get('overall', '?')}\n\n"
        "## Sumário\n"
        f"{toc_text}\n"
        "## Conteúdo por Seção\n"
        f"{section_context}\n\n"
        "---\n"
        "Forneça:\n"
        "### 1. RESUMO EXECUTIVO (3-5 parágrafos)\n"
        "Propósito, escopo, achados principais e conclusões.\n\n"
        "### 2. PONTOS-CHAVE (8-15 itens substantivos)\n\n"
        "### 3. ANÁLISE POR SEÇÃO\n"
        "Para cada seção: tese, argumentos, dados, conclusão parcial.\n\n"
        "### 4. ENTIDADES E REFERÊNCIAS\n"
        "Pessoas, organizações, valores, datas relevantes.\n\n"
        "### 5. QUALIDADE E COMPLETUDE\n"
        "Coerência, lacunas, sugestões."
    )

    try:
        analysis = await llm.chat([
            {"role": "system", "content": "Você é um analista de documentos sênior."},
            {"role": "user", "content": analysis_prompt},
        ])
    except Exception as e:
        analysis = (
            f"Erro ao gerar análise detalhada: {e}\n\n"
            f"## Estrutura encontrada\n{toc_text}\n"
            f"## Estatísticas\n"
            f"- Páginas: {num_pages}\n"
            f"- Blocos: {len(text_blocks)}\n"
            f"- Seções: {len(headings)}\n"
            f"- Tabelas: {len(tables)}\n"
            f"- Qualidade: {quality.get('overall', '?')}\n"
        )

    return {
        "analysis": analysis,
        "structure": {
            "toc": toc,
            "sections_count": len(sections),
            "classification": classified,
            "quality": quality,
            "metadata": metadata,
        },
        "stats": {
            "pages": num_pages,
            "blocks": len(text_blocks),
            "headings": len(headings),
            "tables": len(tables),
        },
    }


# ── Extract structure ────────────────────────────────────────────

_BOILERPLATE_HEADINGS = frozenset({
    "fullscreen", "copy link", "continue reading", "link",
    "continue from your last visit",
})


async def extract_structure(
    extraction_result: dict,
    llm: LLMProvider | None = None,
) -> dict:
    headings = extraction_result.get("headings", [])
    blocks = extraction_result.get("blocks", [])
    tables = extraction_result.get("tables", [])
    metadata = extraction_result.get("metadata", {})
    quality = extraction_result.get("quality", {})
    annotations = extraction_result.get("annotations", [])
    links = extraction_result.get("links", [])

    toc = []
    for h in headings:
        ht = h.get("text", "").strip().split("\n")[0][:80]
        if ht.lower() in _BOILERPLATE_HEADINGS:
            continue
        toc.append({
            "level": h.get("level", 0),
            "text": ht,
            "page": h.get("page", 0),
        })

    sections = []
    current_section = None
    for b in blocks:
        if b.get("type") != "text":
            continue
        is_heading = b.get("is_heading") or b.get("layout_type") == "heading"
        if is_heading:
            if current_section:
                sections.append(current_section)
            ht = b.get("text", "").strip().split("\n")[0][:100]
            if ht.lower() in _BOILERPLATE_HEADINGS:
                current_section = None
                continue
            current_section = {
                "heading": ht,
                "level": b.get("heading_level", 4),
                "page": b.get("page", 0),
                "char_count": 0,
                "block_count": 0,
            }
        elif current_section:
            current_section["char_count"] += len(b.get("text", ""))
            current_section["block_count"] += 1
    if current_section:
        sections.append(current_section)

    classified = {}
    for b in blocks:
        if b.get("type") == "text":
            lt = b.get("layout_type", "unknown")
            classified[lt] = classified.get(lt, 0) + 1

    fonts = {}
    for b in blocks:
        if b.get("type") == "text" and b.get("font"):
            f = b["font"]
            fonts[f] = fonts.get(f, 0) + 1

    table_summaries = []
    for i, t in enumerate(tables[:20]):
        lines = t.strip().split("\n")
        table_summaries.append({
            "index": i,
            "rows": len(lines),
            "preview": t[:200],
        })

    return {
        "toc": toc,
        "sections": sections,
        "classification": classified,
        "fonts": dict(sorted(fonts.items(), key=lambda x: -x[1])[:10]),
        "tables": table_summaries,
        "quality": quality,
        "metadata": metadata,
        "annotations_count": len(annotations),
        "links_count": len(links),
    }


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
    {
        "type": "function",
        "function": {
            "name": "analyze_document",
            "description": (
                "Análise PROFUNDA e estruturada do documento. Use quando o usuário "
                "pedir análise detalhada, avaliação crítica, pontos-chave, análise "
                "por seção, ou qualquer análise que vá além de um simples resumo. "
                "Esta ferramenta retorna: resumo executivo, pontos-chave, análise "
                "por seção, entidades/referências, e avaliação de qualidade."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_structure",
            "description": (
                "Extrai a estrutura completa do documento: sumário (TOC), "
                "seções com estatísticas, classificação de blocos, fontes "
                "utilizadas, tabelas, qualidade e metadados. Use quando o "
                "usuário pedir a estrutura, organização, sumário, ou "
                "visão geral do documento."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
