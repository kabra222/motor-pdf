from __future__ import annotations

import re

from app.models import Block, FormatType, HeadingInfo


def format_result(
    text: str,
    blocks: list[Block],
    headings: list[HeadingInfo],
    tables: list[str],
    fmt: FormatType,
) -> str:
    match fmt:
        case FormatType.markdown:
            return _to_markdown(blocks, headings, tables)
        case FormatType.semantic_html:
            return _to_semantic_html(blocks, headings, tables)
        case _:
            return text


def _detect_footnotes(blocks: list[dict]) -> list[str]:
    notes: list[str] = []
    for b in blocks:
        text = b.get("text", "").strip()
        if re.match(r"^\[\d+\]\s+|^\d+\.\s{2,}", text) and len(text) < 300:
            notes.append(text)
    return notes


def _detect_references(blocks: list[dict]) -> list[str]:
    refs: list[str] = []
    in_refs = False
    for b in blocks:
        text = b.get("text", "").strip()
        if re.match(r"^(refer[eê]ncias|bibliografia|references|bibliography)",
                     text, re.IGNORECASE):
            in_refs = True
            continue
        if in_refs and re.match(r"^[A-Z\u00c0-\u0242][\w\u00c0-\u0242]+.*\(\d{4}\)", text):
            refs.append(text)
    return refs


def _to_markdown(
    blocks: list[Block],
    headings: list[HeadingInfo],
    tables: list[str],
) -> str:
    lines: list[str] = []
    table_idx = 0
    list_buffer: list[tuple[str, str, int]] = []
    blockquote = False
    code_block = False

    def _flush_list():
        if not list_buffer:
            return
        for text, list_type, depth in list_buffer:
            indent = "  " * depth
            marker = "- " if list_type == "bullet" else "1. "
            lines.append(f"{indent}{marker}{text}")
        list_buffer.clear()

    for block in blocks:
        block_dict = block if isinstance(block, dict) else block.model_dump() if hasattr(block, 'model_dump') else {}
        text = getattr(block, 'text', block_dict.get('text', ''))
        is_heading = getattr(block, 'is_heading', block_dict.get('is_heading', False))
        heading_level = getattr(block, 'heading_level', block_dict.get('heading_level'))
        is_list_item = getattr(block, 'is_list_item', block_dict.get('is_list_item', False))
        list_type = getattr(block, 'list_type', block_dict.get('list_type', 'bullet'))
        block_type = getattr(block, 'type', block_dict.get('type', ''))
        page = getattr(block, 'page', block_dict.get('page', 0))
        layout_type = getattr(block, 'layout_type', block_dict.get('layout_type'))

        match block_type:
            case "text":
                if is_heading:
                    _flush_list()
                    if blockquote:
                        lines.append("")
                        blockquote = False
                    prefix = "#" * (heading_level or 1) + " "
                    lines.append(f"\n{prefix}{text}\n")
                elif is_list_item:
                    list_buffer.append((text, list_type or "bullet", 0))
                elif layout_type == "header" or layout_type == "footer":
                    continue
                elif text.startswith("> ") or text.startswith(">"):
                    _flush_list()
                    lines.append(f"\n{text}\n")
                    blockquote = True
                elif re.match(r"^```", text):
                    _flush_list()
                    lines.append(f"\n{text}\n")
                    code_block = not code_block
                else:
                    _flush_list()
                    if blockquote:
                        lines.append(text)
                    else:
                        lines.append(text)
            case "table":
                _flush_list()
                if blockquote:
                    lines.append("")
                    blockquote = False
                if table_idx < len(tables):
                    lines.append(f"\n{tables[table_idx]}\n")
                    table_idx += 1
            case "image":
                _flush_list()
                if blockquote:
                    lines.append("")
                    blockquote = False
                lines.append(f"\n![Imagem - p.{page}]()\n")

    _flush_list()
    result = "\n\n".join(lines).strip()
    footnotes = _detect_footnotes([b if isinstance(b, dict) else b.model_dump() for b in blocks])
    refs = _detect_references([b if isinstance(b, dict) else b.model_dump() for b in blocks])
    if footnotes:
        result += "\n\n---\n**Notas de rodapé:**\n" + "\n".join(f"- {n}" for n in footnotes)
    if refs:
        result += "\n\n---\n**Referências:**\n" + "\n".join(f"- {r}" for r in refs)
    return result


def _to_semantic_html(
    blocks: list[Block],
    headings: list[HeadingInfo],
    tables: list[str],
) -> str:
    parts: list[str] = ['<article class="document">']
    table_idx = 0
    list_buffer: list[tuple[str, str, int]] = []

    def _flush_list():
        if not list_buffer:
            return
        style = "ul" if list_buffer[0][1] == "bullet" else "ol"
        parts.append(f"<{style}>")
        for text, _, _ in list_buffer:
            parts.append(f"<li>{_escape_html(text)}</li>")
        parts.append(f"</{style}>")
        list_buffer.clear()

    for block in blocks:
        block_dict = block if isinstance(block, dict) else block.model_dump() if hasattr(block, 'model_dump') else {}
        text = getattr(block, 'text', block_dict.get('text', ''))
        is_heading = getattr(block, 'is_heading', block_dict.get('is_heading', False))
        heading_level = getattr(block, 'heading_level', block_dict.get('heading_level'))
        is_list_item = getattr(block, 'is_list_item', block_dict.get('is_list_item', False))
        list_type = getattr(block, 'list_type', block_dict.get('list_type', 'bullet'))
        block_type = getattr(block, 'type', block_dict.get('type', ''))
        page = getattr(block, 'page', block_dict.get('page', 0))
        layout_type = getattr(block, 'layout_type', block_dict.get('layout_type'))

        match block_type:
            case "text":
                if is_heading:
                    _flush_list()
                    level = heading_level or 1
                    parts.append(
                        f'<h{level} data-page="{page}">'
                        f"{_escape_html(text)}</h{level}>"
                    )
                elif is_list_item:
                    list_buffer.append((text, list_type or "bullet", 0))
                elif layout_type in ("header", "footer"):
                    continue
                elif text.startswith("> "):
                    _flush_list()
                    parts.append(f'<blockquote data-page="{page}"><p>{_escape_html(text[2:])}</p></blockquote>')
                elif re.match(r"^```", text):
                    _flush_list()
                    parts.append(f'<pre><code data-page="{page}">{_escape_html(text.strip("`"))}</code></pre>')
                else:
                    _flush_list()
                    parts.append(
                        f'<p data-page="{page}">{_escape_html(text)}</p>'
                    )
            case "table":
                _flush_list()
                if table_idx < len(tables):
                    parts.append(
                        f'<figure data-page="{page}">'
                        f"<table>{_table_to_html(tables[table_idx])}</table>"
                        f"</figure>"
                    )
                    table_idx += 1
            case "image":
                _flush_list()
                parts.append(
                    f'<figure data-page="{page}">'
                    f'<img src="" alt="Imagem p.{page}" '
                    f'width="{getattr(block, "width", block_dict.get("width", 0))}" '
                    f'height="{getattr(block, "height", block_dict.get("height", 0))}"/>'
                    f"</figure>"
                )

    _flush_list()
    footnotes = _detect_footnotes([b if isinstance(b, dict) else b.model_dump() for b in blocks])
    refs = _detect_references([b if isinstance(b, dict) else b.model_dump() for b in blocks])
    if footnotes:
        parts.append('<hr/><section class="footnotes">')
        parts.append("<h2>Notas de rodapé</h2>")
        for n in footnotes:
            parts.append(f"<p>{_escape_html(n)}</p>")
        parts.append("</section>")
    if refs:
        parts.append('<hr/><section class="references">')
        parts.append("<h2>Referências</h2>")
        for r in refs:
            parts.append(f"<p>{_escape_html(r)}</p>")
        parts.append("</section>")
    parts.append("</article>")
    return "\n".join(parts)


def _table_to_html(md_table: str) -> str:
    rows = md_table.strip().split("\n")
    if not rows:
        return ""
    html_rows: list[str] = []
    for i, row in enumerate(rows):
        if row.startswith("|") and "---" in row:
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        html_rows.append(
            "<tr>" + "".join(f"<{tag}>{_escape_html(c)}</{tag}>" for c in cells) + "</tr>"
        )
    if len(html_rows) > 1:
        return "<thead>" + html_rows[0] + "</thead><tbody>" + "".join(html_rows[1:]) + "</tbody>"
    return "<tbody>" + "".join(html_rows) + "</tbody>"


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
