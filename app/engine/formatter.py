from __future__ import annotations

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


def _to_markdown(
    blocks: list[Block],
    headings: list[HeadingInfo],
    tables: list[str],
) -> str:
    lines: list[str] = []
    table_idx = 0
    list_buffer: list[tuple[str, str]] = []

    def _flush_list():
        if not list_buffer:
            return
        marker = "- " if list_buffer[0][1] == "bullet" else "1. "
        for text, _ in list_buffer:
            lines.append(f"{marker}{text}")
        list_buffer.clear()

    for block in blocks:
        match block.type:
            case "text":
                if block.is_heading:
                    _flush_list()
                    prefix = "#" * (block.heading_level or 1) + " "
                    lines.append(f"\n{prefix}{block.text}\n")
                elif block.is_list_item:
                    list_buffer.append((block.text, block.list_type or "bullet"))
                else:
                    _flush_list()
                    lines.append(block.text)
            case "table":
                _flush_list()
                if table_idx < len(tables):
                    lines.append(f"\n{tables[table_idx]}\n")
                    table_idx += 1
            case "image":
                _flush_list()
                lines.append(f"\n![Imagem - p.{block.page}]()\n")

    _flush_list()
    return "\n\n".join(lines).strip()


def _to_semantic_html(
    blocks: list[Block],
    headings: list[HeadingInfo],
    tables: list[str],
) -> str:
    parts: list[str] = ['<article class="document">']
    table_idx = 0
    list_buffer: list[tuple[str, str]] = []

    def _flush_list():
        if not list_buffer:
            return
        style = "ul" if list_buffer[0][1] == "bullet" else "ol"
        parts.append(f"<{style}>")
        for text, _ in list_buffer:
            parts.append(f"<li>{_escape_html(text)}</li>")
        parts.append(f"</{style}>")
        list_buffer.clear()

    for block in blocks:
        match block.type:
            case "text":
                if block.is_heading:
                    _flush_list()
                    level = block.heading_level or 1
                    parts.append(
                        f'<h{level} data-page="{block.page}">'
                        f"{_escape_html(block.text)}</h{level}>"
                    )
                elif block.is_list_item:
                    list_buffer.append((block.text, block.list_type or "bullet"))
                else:
                    _flush_list()
                    parts.append(
                        f'<p data-page="{block.page}">{_escape_html(block.text)}</p>'
                    )
            case "table":
                _flush_list()
                if table_idx < len(tables):
                    parts.append(
                        f'<figure data-page="{block.page}">'
                        f"<table>{_table_to_html(tables[table_idx])}</table>"
                        f"</figure>"
                    )
                    table_idx += 1
            case "image":
                _flush_list()
                parts.append(
                    f'<figure data-page="{block.page}">'
                    f'<img src="" alt="Imagem p.{block.page}" '
                    f'width="{block.width}" height="{block.height}"/>'
                    f"</figure>"
                )

    _flush_list()
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
