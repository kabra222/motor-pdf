from __future__ import annotations


def score_extraction(result: dict) -> dict:
    num_pages = result.get("num_pages", 0) or 1
    blocks = result.get("blocks", [])
    headings = result.get("headings", [])
    tables = result.get("tables", [])
    scanned = result.get("scanned_pages", [])
    text = result.get("text", "")

    pages_with_text: set[int] = set()
    text_chars_per_page: dict[int, int] = {}
    for b in blocks:
        if b.get("type") == "text":
            p = b["page"]
            pages_with_text.add(p)
            text_chars_per_page[p] = (
                text_chars_per_page.get(p, 0) + len(b.get("text", ""))
            )

    text_coverage = len(pages_with_text) / max(num_pages, 1)
    text_coverage = min(text_coverage, 1.0)

    has_any_heading = len(headings) > 0
    avg_text_per_page = (
        sum(text_chars_per_page.values()) / max(len(pages_with_text), 1)
        if pages_with_text
        else 0
    )

    scanned_pct = len(scanned) / max(num_pages, 1)

    dims: list[tuple[str, float, str]] = [
        (
            "text_coverage",
            round(text_coverage, 2),
            f"{len(pages_with_text)}/{num_pages} páginas com texto extraído",
        ),
    ]

    if has_any_heading:
        levels = set(h["level"] for h in headings)
        dims.append((
            "heading_detection",
            min(1.0, len(headings) / max(num_pages, 1) * 2),
            f"{len(headings)} headings detectados (níveis: {sorted(levels)})",
        ))
    else:
        dims.append(("heading_detection", 0.0, "Nenhum heading detectado"))

    dims.append((
        "table_extraction",
        1.0 if tables else 0.0,
        f"{len(tables)} tabelas extraídas",
    ))

    dims.append((
        "ocr_required",
        round(1.0 - scanned_pct, 2),
        f"{len(scanned)} páginas com OCR",
    ))

    richness = min(1.0, avg_text_per_page / 500)
    dims.append((
        "text_richness",
        round(richness, 2),
        f"Média de {int(avg_text_per_page)} caracteres por página",
    ))

    overall = round(sum(s for _, s, _ in dims) / len(dims), 2)

    return {
        "overall": overall,
        "dimensions": {
            name: {"score": score, "detail": detail}
            for name, score, detail in dims
        },
    }
