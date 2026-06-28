from __future__ import annotations

from typing import Optional

from app.models import Chunk

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    if HAS_TIKTOKEN:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    return max(1, len(text) // 3)


def _get_tail(text: str, target_tokens: int, model: str) -> str:
    if not text:
        return ""
    total = _count_tokens(text, model)
    if total <= target_tokens:
        return text
    ratio = target_tokens / total
    cutoff = int(len(text) * ratio)
    return text[-cutoff:].lstrip()


def chunk_text(
    text: str,
    pages_text: list[str] | None = None,
    blocks: list[dict] | None = None,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    model: str = "gpt-4",
) -> list[Chunk]:
    if blocks:
        return _semantic_chunk(blocks, chunk_size, chunk_overlap, model)
    return _fallback_chunk(text, pages_text or [text], chunk_size, chunk_overlap, model)


def _semantic_chunk(
    blocks: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    model: str,
) -> list[Chunk]:
    sections: list[dict] = []
    current: dict = {
        "heading": None,
        "level": 0,
        "text_parts": [],
        "pages": set(),
    }

    for block in blocks:
        if block["type"] == "text" and block.get("is_heading"):
            if current["text_parts"] or current["heading"] is not None:
                sections.append(current)
            current = {
                "heading": block["text"],
                "level": block.get("heading_level", 1),
                "text_parts": [],
                "pages": {block["page"]},
            }
        elif block["type"] == "text" and block.get("text", "").strip():
            current["text_parts"].append(block["text"])
            current["pages"].add(block["page"])
        elif block["type"] == "table":
            md = block.get("markdown", "")
            if md:
                current["text_parts"].append(f"\n[Tabela]\n{md}\n")
                current["pages"].add(block["page"])

    if current["text_parts"] or current["heading"] is not None:
        sections.append(current)

    raw_chunks: list[dict] = []
    for section in sections:
        header = (
            f"{'#' * section['level']} {section['heading']}\n"
            if section["heading"]
            else ""
        )
        body = "\n".join(section["text_parts"])
        section_text = (header + body).strip()
        if not section_text:
            continue

        tokens = _count_tokens(section_text, model)
        page = min(section["pages"]) if section["pages"] else 0

        if tokens <= chunk_size:
            raw_chunks.append({
                "text": section_text,
                "page": page,
                "heading": section["heading"],
                "section": section["heading"],
                "tokens": tokens,
            })
        else:
            sub = _split_section(
                section_text, section, chunk_size, chunk_overlap, model
            )
            raw_chunks.extend(sub)

    result: list[Chunk] = []
    char_pos = 0
    for i, c in enumerate(raw_chunks):
        result.append(Chunk(
            index=i,
            text=c["text"],
            page=c["page"],
            section=c.get("section"),
            heading=c.get("heading"),
            tokens=c.get("tokens", 0),
            start_char=char_pos,
            end_char=char_pos + len(c["text"]),
        ))
        char_pos += len(c["text"])

    return result


def _split_section(
    section_text: str,
    section: dict,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
) -> list[dict]:
    paragraphs = section_text.split("\n\n")
    chunks: list[dict] = []
    buffer = ""
    page = min(section["pages"]) if section["pages"] else 0

    for para in paragraphs:
        para_tokens = _count_tokens(para, model)
        buffer_tokens = _count_tokens(buffer, model)

        if not buffer:
            buffer = para
        elif buffer_tokens + para_tokens <= chunk_size:
            buffer += "\n\n" + para
        else:
            chunks.append({
                "text": buffer.strip(),
                "page": page,
                "heading": section["heading"],
                "section": section["heading"],
                "tokens": buffer_tokens,
            })
            if chunk_overlap > 0 and buffer:
                overlap = _get_tail(buffer, chunk_overlap, model)
                buffer = (overlap + "\n\n" + para).strip()
            else:
                buffer = para

    if buffer.strip():
        chunks.append({
            "text": buffer.strip(),
            "page": page,
            "heading": section["heading"],
            "section": section["heading"],
            "tokens": _count_tokens(buffer, model),
        })

    return chunks


def _fallback_chunk(
    text: str,
    pages_text: list[str],
    chunk_size: int,
    chunk_overlap: int,
    model: str,
) -> list[Chunk]:
    page_offsets: list[int] = []
    offset = 0
    for pt in pages_text:
        page_offsets.append(offset)
        offset += len(pt) + 1

    def _find_page(char_pos: int) -> int:
        for i in range(len(page_offsets)):
            if i + 1 < len(page_offsets) and page_offsets[i] <= char_pos < page_offsets[i + 1]:
                return i
        return len(page_offsets) - 1 if page_offsets else 0

    paragraphs = text.split("\n\n")
    raw: list[dict] = []
    buffer = ""
    start_char = 0

    for para in paragraphs:
        while _count_tokens(para, model) > chunk_size:
            part = para[:chunk_size]
            raw.append({
                "text": part,
                "page": _find_page(start_char),
                "heading": None,
                "section": None,
                "tokens": _count_tokens(part, model),
            })
            para = para[chunk_size - chunk_overlap:]
            start_char += len(part) - chunk_overlap

        candidate = buffer + "\n\n" + para if buffer else para
        buffer_tokens = _count_tokens(candidate, model)

        if buffer_tokens > chunk_size and buffer:
            raw.append({
                "text": buffer,
                "page": _find_page(start_char),
                "heading": None,
                "section": None,
                "tokens": _count_tokens(buffer, model),
            })
            start_char += len(buffer) - chunk_overlap
            if chunk_overlap > 0:
                overlap_text = _get_tail(buffer, chunk_overlap, model)
                buffer = (overlap_text + "\n\n" + para).strip()
            else:
                buffer = para
        else:
            buffer = candidate

    if buffer.strip():
        raw.append({
            "text": buffer.strip(),
            "page": _find_page(start_char),
            "heading": None,
            "section": None,
            "tokens": _count_tokens(buffer, model),
        })

    result: list[Chunk] = []
    char_pos = 0
    for i, c in enumerate(raw):
        result.append(Chunk(
            index=i,
            text=c["text"],
            page=c["page"],
            section=c.get("section"),
            heading=c.get("heading"),
            tokens=c.get("tokens", 0),
            start_char=char_pos,
            end_char=char_pos + len(c["text"]),
        ))
        char_pos += len(c["text"])

    return result
