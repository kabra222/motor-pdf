from __future__ import annotations

import re

from app.models import Chunk

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from app.engine.segmenter import Segment, SegmentationResult, segment_hybrid
    HAS_SEGMENTER = True
except ImportError:
    HAS_SEGMENTER = False

MIN_CHUNK_TOKENS = 50


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


def _dedup_chunks(chunks: list[dict], model: str) -> list[dict]:
    if len(chunks) < 2:
        return chunks
    deduped: list[dict] = [chunks[0]]
    for c in chunks[1:]:
        prev_text = deduped[-1]["text"]
        curr_text = c["text"]
        if curr_text in prev_text or prev_text in curr_text:
            if len(curr_text) > len(prev_text):
                deduped[-1] = c
            continue
        prev_lines = set(prev_text.split("\n"))
        curr_lines = curr_text.split("\n")
        new_lines = [l for l in curr_lines if l not in prev_lines or len(l.strip()) > 80]
        if len(new_lines) < len(curr_lines) * 0.3:
            continue
        c["text"] = "\n".join(new_lines)
        c["tokens"] = _count_tokens(c["text"], model)
        if c["tokens"] >= MIN_CHUNK_TOKENS:
            deduped.append(c)
    return deduped


def _merge_tiny_chunks(chunks: list[dict], model: str) -> list[dict]:
    if not chunks:
        return chunks
    merged: list[dict] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if c["tokens"] < MIN_CHUNK_TOKENS and i + 1 < len(chunks):
            next_c = chunks[i + 1]
            combined_text = c["text"] + "\n\n" + next_c["text"]
            combined_tokens = _count_tokens(combined_text, model)
            if combined_tokens <= next_c.get("max_tokens", 999999):
                merged.append({
                    "text": combined_text,
                    "page": c["page"],
                    "heading": c.get("heading") or next_c.get("heading"),
                    "section": c.get("section") or next_c.get("section"),
                    "depth": c.get("depth", 0),
                    "tokens": combined_tokens,
                })
                i += 2
                continue
        merged.append(c)
        i += 1
    return merged


def _is_mostly_formatting(text: str) -> bool:
    lines = text.strip().split("\n")
    if not lines:
        return True
    formatting_lines = sum(
        1 for l in lines
        if re.match(r"^\s*[■•\-\*]\s*$", l)
        or re.match(r"^\s*#{1,6}\s*$", l)
        or re.match(r"^\s*\|.*\|\s*$", l)
        or len(l.strip()) < 5
    )
    return formatting_lines > len(lines) * 0.6


def chunk_text(
    text: str,
    pages_text: list[str] | None = None,
    blocks: list[dict] | None = None,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    model: str = "gpt-4",
    use_bcpd: bool = False,
) -> list[Chunk]:
    if use_bcpd and HAS_SEGMENTER and blocks:
        seg_result = segment_hybrid(text, blocks, chunk_size)
        if seg_result.segments:
            chunks = _segments_to_chunks(seg_result.segments, chunk_size, chunk_overlap, model)
            return _postprocess(chunks, model)
    if blocks:
        chunks = _semantic_chunk(blocks, chunk_size, chunk_overlap, model)
        return _postprocess(chunks, model)
    chunks = _fallback_chunk(text, pages_text or [text], chunk_size, chunk_overlap, model)
    return _postprocess(chunks, model)


def _postprocess(chunks: list[Chunk], model: str) -> list[Chunk]:
    raw = [
        {
            "text": c.text,
            "page": c.page,
            "heading": c.heading,
            "section": c.section,
            "depth": c.depth,
            "tokens": c.tokens,
        }
        for c in chunks
    ]
    raw = _merge_tiny_chunks(raw, model)
    raw = _dedup_chunks(raw, model)
    raw = [c for c in raw if not _is_mostly_formatting(c["text"])]
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
            depth=c.get("depth", 0),
            start_char=char_pos,
            end_char=char_pos + len(c["text"]),
        ))
        char_pos += len(c["text"])
    return result


def _segments_to_chunks(
    segments: list[Segment],
    chunk_size: int,
    chunk_overlap: int,
    model: str,
) -> list[Chunk]:
    raw: list[dict] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        tokens = _count_tokens(text, model)
        if tokens <= chunk_size:
            raw.append({
                "text": text,
                "page": seg.page,
                "heading": seg.heading,
                "section": seg.heading,
                "depth": seg.depth,
                "tokens": tokens,
            })
        else:
            sub = _split_by_tokens(text, seg, chunk_size, chunk_overlap, model)
            raw.extend(sub)

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
            depth=c.get("depth", 0),
            start_char=char_pos,
            end_char=char_pos + len(c["text"]),
        ))
        char_pos += len(c["text"])
    return result


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
            ht = block["text"].strip().split("\n")[0][:100]
            current = {
                "heading": ht,
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
                "depth": section["level"],
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
            depth=c.get("depth", 0),
            start_char=char_pos,
            end_char=char_pos + len(c["text"]),
        ))
        char_pos += len(c["text"])

    return result


def _split_by_tokens(
    text: str,
    seg: Segment,
    chunk_size: int,
    chunk_overlap: int,
    model: str,
) -> list[dict]:
    paragraphs = text.split("\n\n")
    chunks: list[dict] = []
    buffer = ""
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
                "page": seg.page,
                "heading": seg.heading,
                "section": seg.heading,
                "depth": seg.depth,
                "tokens": buffer_tokens,
            })
            buffer = para
    if buffer.strip():
        chunks.append({
            "text": buffer.strip(),
            "page": seg.page,
            "heading": seg.heading,
            "section": seg.heading,
            "depth": seg.depth,
            "tokens": _count_tokens(buffer, model),
        })
    return chunks


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
                "depth": section["level"],
                "tokens": buffer_tokens,
            })
            buffer = para

    if buffer.strip():
        chunks.append({
            "text": buffer.strip(),
            "page": page,
            "heading": section["heading"],
            "section": section["heading"],
            "depth": section["level"],
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
