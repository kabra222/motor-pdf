from __future__ import annotations

import json
import re

from app.agent.llm import LLMProvider


async def mine_arguments(
    extraction_result: dict,
    llm: LLMProvider,
) -> dict:
    headings = extraction_result.get("headings", [])
    blocks = extraction_result.get("blocks", [])
    text = extraction_result.get("text", "")
    num_pages = extraction_result.get("num_pages", 0)

    sections = _build_sections(blocks)
    section_summaries = []
    char_budget = 10000
    used = 0
    for sec in sections[:25]:
        content = sec["content"].strip()
        if len(content) < 30:
            continue
        chunk = f"## {sec['heading']}\n{content[:600]}\n\n"
        if used + len(chunk) > char_budget:
            break
        section_summaries.append(chunk)
        used += len(chunk)

    sections_text = "".join(section_summaries)

    toc = []
    for h in headings[:30]:
        ht = h.get("text", "").strip().split("\n")[0][:80]
        toc.append(f"- {ht}")

    prompt = (
        "Você é um especialista em análise argumentativa e mineração de estruturas "
        "de texto. Analise o documento abaixo e extraia a estrutura argumentativa "
        "e temática completa.\n\n"
        f"Documento ({num_pages} páginas):\n\n"
        f"## Sumário\n{'chr(10)'.join(toc)}\n\n"
        f"## Conteúdo por Seção\n{sections_text}\n\n"
        "---\n\n"
        "Retorne APENAS um JSON válido com esta estrutura exata:\n"
        "{\n"
        '  "titulo": "título principal do documento",\n'
        '  "tese_central": "tese/argumento central em 1-2 frases",\n'
        '  "topicos": [\n'
        "    {\n"
        '      "nome": "nome do tópico",\n'
        '      "tipo": "conceito|argumento|exemplo|contra-argumento|conclusao",\n'
        '      "descricao": "resumo em 1-2 frases",\n'
        '      "subtopicos": [\n'
        "        {\n"
        '          "nome": "subtópico",\n'
        '          "tipo": "conceito|argumento|exemplo|contra-argumento|conclusao",\n'
        '          "descricao": "resumo"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "relacoes": [\n'
        "    {\n"
        '      "de": "nome do tópico origem",\n'
        '      "para": "nome do tópico destino",\n'
        '      "tipo": "suporta|contrasta|exemplifica|conclui"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Regras:\n"
        "- Extraia 5-12 tópicos principais\n"
        "- Cada tópico pode ter 0-5 subtópicos\n"
        "- Use os tipos corretamente:\n"
        "  - conceito: definições, classificações, teorias\n"
        "  - argumento: teses, posições, justificativas\n"
        "  - exemplo: casos, dados, evidências\n"
        "  - contra-argumento: objeções, críticas\n"
        "  - conclusao: sínteses, consequências\n"
        "- Inclua 3-8 relações entre tópicos\n"
        "- Seja específico e substantivo, não trivial\n"
        "- Responda APENAS com o JSON, sem markdown"
    )

    try:
        result = await llm.chat([
            {"role": "system", "content": "Você é um analista argumentativo especializado."},
            {"role": "user", "content": prompt},
        ])
    except Exception as e:
        return {
            "error": f"Erro na análise argumentativa: {e}",
            "fallback_markdown": _fallback_mindmap(headings),
        }

    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```(?:json)?\s*", "", result)
        result = re.sub(r"\s*```$", "", result)

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {
                    "error": "Falha ao parsear resposta",
                    "raw": result[:500],
                    "fallback_markdown": _fallback_mindmap(headings),
                }
        else:
            return {
                "error": "Resposta não contém JSON",
                "raw": result[:500],
                "fallback_markdown": _fallback_mindmap(headings),
            }

    markdown = _to_markmap_markdown(data)
    data["markmap_markdown"] = markdown
    return data


def _build_sections(blocks: list[dict]) -> list[dict]:
    sections: list[dict] = []
    current: dict | None = None
    for b in blocks:
        if b.get("type") != "text":
            continue
        is_heading = b.get("is_heading") or b.get("layout_type") == "heading"
        if is_heading:
            if current:
                sections.append(current)
            ht = b.get("text", "").strip().split("\n")[0][:100]
            current = {
                "heading": ht,
                "level": b.get("heading_level", 4),
                "page": b.get("page", 0),
                "content": "",
            }
        elif current is not None:
            current["content"] += b.get("text", "") + "\n"
        else:
            current = {
                "heading": "(Pré-texto)",
                "level": 0,
                "page": 0,
                "content": b.get("text", "") + "\n",
            }
    if current:
        sections.append(current)
    return sections


def _to_markmap_markdown(data: dict) -> str:
    lines: list[str] = []
    titulo = data.get("titulo", "Documento")
    tese = data.get("tese_central", "")
    lines.append(f"# {titulo}")
    if tese:
        lines.append(f"## Tese Central")
        lines.append(f"### {tese}")

    tipo_icons = {
        "conceito": "📘",
        "argumento": "⚡",
        "exemplo": "📌",
        "contra-argumento": "⚠️",
        "conclusao": "✅",
    }

    for topico in data.get("topicos", []):
        nome = topico.get("nome", "Tópico")
        tipo = topico.get("tipo", "conceito")
        desc = topico.get("descricao", "")
        icon = tipo_icons.get(tipo, "📄")
        lines.append(f"## {icon} {nome}")
        if desc:
            lines.append(f"### {desc}")
        for sub in topico.get("subtopicos", []):
            sub_nome = sub.get("nome", "")
            sub_tipo = sub.get("tipo", "conceito")
            sub_desc = sub.get("descricao", "")
            sub_icon = tipo_icons.get(sub_tipo, "📄")
            lines.append(f"### {sub_icon} {sub_nome}")
            if sub_desc:
                lines.append(f"#### {sub_desc}")

    relacoes = data.get("relacoes", [])
    if relacoes:
        lines.append("## 🔗 Relações Argumentativas")
        for r in relacoes:
            de = r.get("de", "?")
            para = r.get("para", "?")
            tipo_r = r.get("tipo", "suporta")
            arrow = {"suporta": "→", "contrasta": "⇝", "exemplifica": "⇒", "conclui": "⇛"}.get(tipo_r, "→")
            lines.append(f"### {de} {arrow} {para} ({tipo_r})")

    return "\n".join(lines)


def _fallback_mindmap(headings: list[dict]) -> str:
    lines = ["# Documento"]
    for h in headings[:30]:
        level = h.get("level", 4)
        ht = h.get("text", "").strip().split("\n")[0][:80]
        prefix = "#" * min(level + 1, 6)
        lines.append(f"{prefix} {ht}")
    return "\n".join(lines)
