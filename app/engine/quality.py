from __future__ import annotations

import re

_HYPHEN_PATTERN = re.compile(
    r"(\w[-\w]*\w)-\n\s*(\w[-\w]*\w)", re.UNICODE
)
_ORPHAN_LINE = re.compile(r"^\s*\w{1,3}\s*$", re.UNICODE)


def score_extraction(result: dict) -> dict:
    num_pages = result.get("num_pages", 0) or 1
    blocks = result.get("blocks", [])
    headings = result.get("headings", [])
    tables = result.get("tables", [])
    scanned = result.get("scanned_pages", [])
    text = result.get("text", "")

    dims: list[tuple[str, float, str]] = []

    text_lines = text.split("\n")
    total_lines = len(text_lines) if text_lines else 1

    hyphen_broken = sum(1 for _ in _HYPHEN_PATTERN.finditer(text))
    hyphen_penalty = min(1.0, hyphen_broken / max(total_lines * 0.02, 1))
    hyphen_score = max(0.0, round(1.0 - hyphen_penalty, 2))
    dims.append((
        "hyphenation",
        hyphen_score,
        f"{hyphen_broken} quebras de hifen nao reparadas",
    ))

    orphan_count = sum(
        1 for line in text_lines
        if _ORPHAN_LINE.match(line)
        and not line.strip().startswith("#")
    )
    orphan_penalty = min(1.0, orphan_count / max(total_lines * 0.05, 1))
    orphan_score = max(0.0, round(1.0 - orphan_penalty, 2))
    dims.append((
        "orphan_lines",
        orphan_score,
        f"{orphan_count} linhas orfas (justificativa)",
    ))

    chars = sum(len(b.get("text", "")) for b in blocks if b.get("type") == "text")
    expected_chars = num_pages * 2000
    if expected_chars > 0:
        coverage = min(1.0, chars / expected_chars)
    else:
        coverage = 1.0
    if coverage > 0:
        coverage = max(0.1, coverage)
    dims.append((
        "text_coverage",
        round(coverage, 2),
        f"{chars} caracteres extraidos ({int(coverage * 100)}% esperado)",
    ))

    if headings:
        levels = sorted(set(h["level"] for h in headings))
        heading_density = min(1.0, len(headings) / max(num_pages * 2, 1))
        dims.append((
            "headings",
            round(heading_density, 2),
            f"{len(headings)} headings detectados (niveis: {levels})",
        ))
    else:
        dims.append(("headings", 0.0, "Nenhum heading detectado"))

    table_score = min(1.0, len(tables) / max(num_pages * 0.1, 1))
    dims.append((
        "tables",
        round(table_score, 2),
        f"{len(tables)} tabelas extraidas",
    ))

    scanned_pct = len(scanned) / max(num_pages, 1)
    dims.append((
        "ocr_quality",
        round(1.0 - scanned_pct, 2),
        f"{len(scanned)} paginas com OCR",
    ))

    avg_line_len = sum(len(l) for l in text_lines) / max(len(text_lines), 1)
    line_quality = min(1.0, max(0.0, avg_line_len / 80))
    dims.append((
        "line_quality",
        round(line_quality, 2),
        f"Media {int(avg_line_len)} caracteres/linha",
    ))

    overall = round(sum(s for _, s, _ in dims) / len(dims), 2)

    return {
        "overall": overall,
        "dimensions": {
            name: {"score": score, "detail": detail}
            for name, score, detail in dims
        },
    }
