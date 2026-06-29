from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HAS_UNSTRUCTURED = False
try:
    from unstructured.partition.pdf import partition_pdf
    HAS_UNSTRUCTURED = True
except ImportError:
    pass


@dataclass
class ClassifiedElement:
    text: str
    type: str
    page: int
    confidence: float = 1.0
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


_NUMERIC_PAGE = re.compile(r"^\s*\d+\s*$")
_PAGE_LABEL = re.compile(
    r"^\s*(p[aá]gina|p[aá]g\.?|fl\.?|folha|page|p\.)\s*\d+",
    re.IGNORECASE,
)
_HEADER_LIKE = re.compile(
    r"^\s*(cap[ií]tulo|se[cç][aã]o|parte|t[ií]tulo|livro)\s+\d+",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:"
    r"\d+\.\d+(?:\.\d+)*(?:\s+[—–‑\-]\s*|\s+)(?=[A-ZÀ-Ú])"     # 9.8 Title / 1.2.3.4 Title
    r"|"
    r"\d+(?:\s+[—–‑\-]\s*|\s+)(?=[A-ZÀ-Ú])(?!.*\d{2,}\s)"     # 1 Title / 2 Background (not table data)
    r"|"
    r"[IVXLCDM]+\.\s+"                                           # I.  II.
    r"|"
    r"[A-Z]\.\s+"                                                 # A.  B.
    r"|"
    r"(?:EXERC[ÍI]CIO|EXERCISE|PROBLEM|QUEST[AÃ]O"
    r"|SECTION|CHAPTER)\s+\d+(?:\.\d+)*"                         # EXERCISE 9.10
    r"|"
    r"(?:APPENDIX|APP[.]|ANEXO)\s+[A-Z0-9]+"                    # APPENDIX A
    r")",
    re.IGNORECASE,
)
_SHORT_UPPER_HEADING = re.compile(
    r"^\s*[A-ZÀ-Ú][A-ZÀ-Ú\s]{2,30}$",
)
_COMMON_HEADING_WORDS = frozenset({
    "abstract", "resumo", "introduction", "introdução",
    "conclusion", "conclusão", "conclusions", "conclusões",
    "references", "referências", "bibliography", "bibliografia",
    "acknowledgments", "acknowledgements", "agradecimentos",
    "appendix", "apêndice", "anexo", "annex",
    "background", "methods", "methodology", "metodologia",
    "results", "resultados", "discussion", "discussão",
    "summary", "resumo", "overview",
})


def classify_blocks_builtin(blocks: list[dict]) -> list[dict]:
    if not blocks:
        return blocks

    page_texts: dict[int, list[str]] = {}
    for b in blocks:
        p = b.get("page", 0)
        if p not in page_texts:
            page_texts[p] = []
        if b.get("type") == "text":
            page_texts[p].append(b.get("text", ""))

    header_candidates: Counter[str] = Counter()
    footer_candidates: Counter[str] = Counter()

    for p in sorted(page_texts.keys()):
        texts = page_texts[p]
        if len(texts) >= 2:
            header_candidates[texts[0].strip()] += 1
            footer_candidates[texts[-1].strip()] += 1

    num_pages = len(page_texts)
    hf_threshold = max(2, num_pages * 0.5)

    likely_headers = {
        t for t, c in header_candidates.most_common(5)
        if c >= hf_threshold and len(t) > 5
    }
    likely_footers = {
        t for t, c in footer_candidates.most_common(5)
        if c >= hf_threshold and len(t) > 5
    }

    enriched: list[dict] = []
    for b in blocks:
        enriched_b = dict(b)
        if b.get("type") == "text":
            text = b.get("text", "").strip()
            label = _classify_block_text(text, enriched_b, likely_headers, likely_footers)
            enriched_b["layout_type"] = label
            enriched_b["is_noise"] = label in ("header", "footer", "page_number")
        enriched.append(enriched_b)

    return enriched


def _infer_heading_level(text: str) -> int | None:
    m = _NUMBERED_HEADING.match(text)
    if not m:
        return None
    prefix = m.group(0).strip()
    if re.match(
        r"^(?:EXERC[ÍI]CIO|EXERCISE|PROBLEM|QUEST[AÃ]O|SECTION|CHAPTER|APPENDIX|ANEXO)",
        prefix,
        re.IGNORECASE,
    ):
        return 1
    if re.match(r"^[IVXLCDM]+\.", prefix):
        return 1
    if re.match(r"^[A-Z]\.", prefix):
        return 2
    dot_count = prefix.count(".")
    if dot_count >= 2:
        return 3
    if dot_count == 1:
        return 2
    return 1


def _classify_block_text(
    text: str,
    block: dict,
    likely_headers: set[str],
    likely_footers: set[str],
) -> str:
    if _NUMERIC_PAGE.match(text):
        return "page_number"
    if _PAGE_LABEL.match(text):
        return "page_number"

    if text in likely_headers:
        return "header"
    if text in likely_footers:
        return "footer"

    single_line = re.sub(r"\s+", " ", text).strip()
    inferred = _infer_heading_level(text) or _infer_heading_level(single_line)
    if inferred is not None:
        block["heading_level"] = inferred
        block["is_heading"] = True
        return "heading"

    if block.get("is_heading"):
        return "heading"

    if _HEADER_LIKE.match(text):
        return "heading"

    if single_line.lower() in _COMMON_HEADING_WORDS:
        block["heading_level"] = 1
        block["is_heading"] = True
        return "heading"

    if block.get("is_list_item"):
        return "list_item"

    if block.get("inside_table"):
        return "table_cell"

    if _SHORT_UPPER_HEADING.match(text) or _SHORT_UPPER_HEADING.match(single_line):
        block["heading_level"] = 2
        block["is_heading"] = True
        return "heading"

    if len(text) < 15:
        return "short_text"

    return "paragraph"


def filter_noise_blocks(
    blocks: list[dict],
    classified: list[ClassifiedElement] | None = None,
) -> list[dict]:
    if classified:
        blocks = _merge_with_external(blocks, classified)
    else:
        blocks = classify_blocks_builtin(blocks)

    cleaned: list[dict] = []
    for b in blocks:
        if b.get("is_noise"):
            continue
        cleaned.append(b)

    return cleaned


def _merge_with_external(
    blocks: list[dict],
    classified: list[ClassifiedElement],
) -> list[dict]:
    header_texts: set[str] = set()
    footer_texts: set[str] = set()
    for el in classified:
        if el.type == "header":
            header_texts.add(el.text.strip())
        elif el.type == "footer":
            footer_texts.add(el.text.strip())

    enriched: list[dict] = []
    for b in blocks:
        enriched_b = dict(b)
        if b.get("type") == "text":
            text = b.get("text", "").strip()
            label = "paragraph"

            for el in classified:
                if el.page == b.get("page", 0) and el.text.strip() == text:
                    label = el.type
                    break

            if text in header_texts:
                label = "header"
            if text in footer_texts:
                label = "footer"
            if b.get("is_heading"):
                label = "heading"

            enriched_b["layout_type"] = label
            enriched_b["is_noise"] = label in ("header", "footer", "page_number")

        enriched.append(enriched_b)

    return enriched


def classify_pdf(
    path: str | Path,
    strategy: str = "auto",
) -> list[ClassifiedElement]:
    if not HAS_UNSTRUCTURED:
        return []

    elements: list[ClassifiedElement] = []
    try:
        raw = partition_pdf(
            str(path),
            strategy=strategy,
            include_page_breaks=True,
        )
    except Exception:
        return []

    _TYPE_MAP = {
        "Title": "heading",
        "Header": "header",
        "Footer": "footer",
        "Table": "table",
        "Figure": "figure",
        "ListItem": "list_item",
        "Formula": "formula",
        "PageBreak": "page_break",
        "FigureCaption": "caption",
        "TableCaption": "caption",
        "NarrativeText": "paragraph",
        "UncategorizedText": "text",
    }

    for el in raw:
        try:
            el_type = _TYPE_MAP.get(el.category, "text")
            page = el.metadata.page_number or 0
            bbox_tuple = (0, 0, 0, 0)
            if el.metadata.coordinates:
                pts = el.metadata.coordinates.points
                if pts and len(pts) >= 2:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    bbox_tuple = (min(xs), min(ys), max(xs), max(ys))

            elements.append(ClassifiedElement(
                text=str(el.text),
                type=el_type,
                page=page,
                confidence=1.0,
                bbox=bbox_tuple,
            ))
        except Exception:
            continue

    return elements


def merge_with_blocks(
    blocks: list[dict],
    classified: list[ClassifiedElement],
) -> list[dict]:
    return _merge_with_external(blocks, classified)


def is_unstructured_available() -> bool:
    return HAS_UNSTRUCTURED
