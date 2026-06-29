from __future__ import annotations

import re

_BOILERPLATE_LINES: list[re.Pattern] = [
    re.compile(r"^\s*Continue from your last visit\??"),
    re.compile(r"^\s*FULLSCREEN\s*$"),
    re.compile(r"^\s*COPY LINK\s*$"),
    re.compile(r"^\s*CONTINUE READING\s*$"),
    re.compile(r"^\s*LINK\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
]

_PAGE_NUMBER_INLINE = re.compile(r"\b\d+\s*/\s*\d+\b")


def is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in _BOILERPLATE_LINES:
        if pattern.match(stripped):
            return True
    return False


def filter_boilerplate(text: str) -> str:
    lines = text.split("\n")
    filtered = [l for l in lines if not is_boilerplate_line(l)]
    return "\n".join(filtered)


def filter_boilerplate_from_blocks(blocks: list[dict]) -> list[dict]:
    return [b for b in blocks if not is_boilerplate_line(b.get("text", ""))]


def remove_page_markers_inline(text: str) -> str:
    return _PAGE_NUMBER_INLINE.sub("", text).strip()
