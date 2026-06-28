from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TableRegion:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    rows: int = 0
    cols: int = 0


@dataclass
class ColumnBoundary:
    x: float
    page: int


@dataclass
class LayoutInfo:
    tables: list[TableRegion] = field(default_factory=list)
    columns: list[ColumnBoundary] = field(default_factory=list)
    header_rect: tuple[float, float, float, float] | None = None
    footer_rect: tuple[float, float, float, float] | None = None
    has_multi_column: bool = False


# ── Blocks analysis for columns, headers/footers ────────────


def analyze_blocks(blocks: list[dict], num_pages: int) -> LayoutInfo:
    info = LayoutInfo()
    if not blocks:
        return info

    page_width = max((b.get("bbox", (0, 0, 0, 0))[2] for b in blocks), default=612)

    left_edges: dict[int, list[float]] = {}
    right_edges: dict[int, list[float]] = {}
    top_edges: dict[int, list[float]] = {}
    bottom_edges: dict[int, list[float]] = {}

    for b in blocks:
        bbox = b.get("bbox", (0, 0, 0, 0))
        p = b.get("page", 0)
        if p not in left_edges:
            left_edges[p] = []
            right_edges[p] = []
            top_edges[p] = []
            bottom_edges[p] = []
        left_edges[p].append(bbox[0])
        right_edges[p].append(bbox[2])
        top_edges[p].append(bbox[1])
        bottom_edges[p].append(bbox[3])

    top_margins: list[float] = []
    bottom_margins: list[float] = []
    for p in range(num_pages):
        if p in top_edges and top_edges[p]:
            top_margins.append(min(top_edges[p]))
        if p in bottom_edges and bottom_edges[p]:
            bottom_margins.append(max(bottom_edges[p]))

    if top_margins:
        median_top = statistics.median(top_margins)
        consistent_top = [t for t in top_margins if abs(t - median_top) < 20]
        if len(consistent_top) > max(2, num_pages * 0.5):
            pass

    if bottom_margins:
        median_bottom = statistics.median(bottom_margins)
        consistent_bottom = [t for t in bottom_margins if abs(t - median_bottom) < 20]
        if len(consistent_bottom) > max(2, num_pages * 0.5):
            info.footer_rect = (0, median_bottom - 30, page_width, median_bottom)

    all_centers: list[float] = []
    for b in blocks:
        bbox = b.get("bbox", (0, 0, 0, 0))
        cx = (bbox[0] + bbox[2]) / 2
        if b.get("type") == "text" and not b.get("inside_table", False):
            all_centers.append(cx)

    if len(all_centers) > 10:
        rounded = [round(c / 20) * 20 for c in all_centers]
        counts = Counter(rounded)
        major_cols = [x for x, c in counts.most_common(3) if c > max(5, len(all_centers) * 0.05)]
        major_cols.sort()

        gap = page_width / 2
        col_groups: list[list[float]] = [[]]
        for col in major_cols:
            if col_groups[-1] and col - col_groups[-1][0] > page_width * 0.3:
                if abs(col - page_width / 2) < page_width * 0.3:
                    col_groups[-1].append(col)
                else:
                    col_groups.append([col])
            else:
                col_groups[-1].append(col)

        if len(col_groups) >= 2:
            info.has_multi_column = True
            for group in col_groups:
                mid = statistics.median(group) if len(group) > 1 else group[0]
                info.columns.append(ColumnBoundary(x=mid, page=-1))

    return info


def detect_headers_footers_text(blocks: list[dict], num_pages: int) -> tuple[set[str], set[str]]:
    page_texts: dict[int, list[str]] = {}
    for b in blocks:
        p = b.get("page", 0)
        if p not in page_texts:
            page_texts[p] = []
        if b.get("type") == "text" and b.get("text", "").strip():
            page_texts[p].append(b["text"])

    header_candidates: Counter[str] = Counter()
    footer_candidates: Counter[str] = Counter()

    for p in range(num_pages):
        texts = page_texts.get(p, [])
        if len(texts) < 2:
            continue
        first_block = texts[0]
        last_block = texts[-1]
        if len(first_block) > 5:
            header_candidates[first_block] += 1
        if len(last_block) > 5:
            footer_candidates[last_block] += 1

    threshold = max(2, num_pages * 0.6)
    headers = {t for t, c in header_candidates.most_common(5) if c >= threshold}
    footers = {t for t, c in footer_candidates.most_common(5) if c >= threshold}

    return headers, footers


def strip_headers_footers_from_blocks(
    blocks: list[dict], num_pages: int
) -> list[dict]:
    headers, footers = detect_headers_footers_text(blocks, num_pages)
    if not headers and not footers:
        return blocks

    cleaned: list[dict] = []
    for b in blocks:
        if b.get("type") == "text":
            text = b.get("text", "").strip()
            if text in headers or text in footers:
                continue
        cleaned.append(b)
    return cleaned


def strip_headers_footers(
    text: str, pages_text: list[str]
) -> str:
    headers, footers = _detect_hf_from_text(pages_text)
    if not headers and not footers:
        return text

    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in headers or stripped in footers:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _detect_hf_from_text(
    pages_text: list[str],
    header_ratio: float = 0.1,
    footer_ratio: float = 0.1,
) -> tuple[set[str], set[str]]:
    num_pages = len(pages_text)
    if num_pages < 3:
        return set(), set()

    header_candidates: Counter[str] = Counter()
    footer_candidates: Counter[str] = Counter()

    for page_text in pages_text:
        lines = page_text.split("\n")
        if not lines:
            continue
        first_n = max(1, int(len(lines) * header_ratio))
        last_n = max(1, int(len(lines) * footer_ratio))
        for line in lines[:first_n]:
            stripped = line.strip()
            if stripped and len(stripped) > 5:
                header_candidates[stripped] += 1
        for line in lines[-last_n:]:
            stripped = line.strip()
            if stripped and len(stripped) > 5:
                footer_candidates[stripped] += 1

    threshold = max(2, num_pages * 0.6)
    headers = {t for t, c in header_candidates.most_common(5) if c >= threshold}
    footers = {t for t, c in footer_candidates.most_common(5) if c >= threshold}

    return headers, footers


# ── Image-based column detection (optional) ─────────────────


def detect_columns_via_projection(
    path: str | Path, page_num: int = 0, dpi: int = 100
) -> list[ColumnBoundary]:
    try:
        from PIL import Image
        from pdf2image import convert_from_path
    except ImportError:
        return []

    try:
        images = convert_from_path(
            str(path), first_page=page_num + 1, last_page=page_num + 1, dpi=dpi
        )
        if not images:
            return []
        img = images[0]

        import numpy as np

        arr = np.array(img.convert("L"))
        binary = (arr < 200).astype(np.uint8)
        proj = binary.sum(axis=0)

        width = img.width
        threshold = max(1, int(width * 0.01))
        blank_threshold = proj.max() * 0.02

        bounds: list[ColumnBoundary] = []
        in_gap = False
        gap_start = 0

        for x in range(width):
            if proj[x] < blank_threshold:
                if not in_gap:
                    gap_start = x
                    in_gap = True
            else:
                if in_gap:
                    gap_width = x - gap_start
                    if gap_width >= threshold and width * 0.1 < (gap_start + gap_width / 2) < width * 0.9:
                        cx = (gap_start + gap_start + gap_width) / 2
                        bounds.append(ColumnBoundary(x=cx * dpi / 72, page=page_num))
                    in_gap = False

        return bounds
    except Exception:
        return []


# ── Table overlap filtering ────────────────────────────────


def is_inside_table(bbox: tuple[float, ...], tables: list[TableRegion]) -> bool:
    if not tables:
        return False
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for t in tables:
        if t.x0 <= cx <= t.x1 and t.y0 <= cy <= t.y1:
            return True
    return False
