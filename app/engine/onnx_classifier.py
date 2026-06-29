from __future__ import annotations

import re
from collections import Counter
from statistics import median
from typing import Any

HAS_ONNX = False
try:
    import onnxruntime  # noqa: F401
    HAS_ONNX = True
except ImportError:
    pass


def classify_blocks_onnx(blocks: list[dict]) -> list[dict]:
    if not blocks:
        return blocks

    text_blocks = [b for b in blocks if b.get("type") == "text"]
    if not text_blocks:
        return blocks

    body_size = _compute_body_size(text_blocks)
    page_heights = _compute_page_heights(blocks)
    repeated_texts = _find_repeated_texts(text_blocks)

    for b in blocks:
        if b.get("type") != "text":
            continue
        if "layout_type" in b and "is_noise" in b:
            continue

        fv = _feature_vector(b, body_size, page_heights, repeated_texts)
        layout_type, is_noise = _classify(fv, body_size, b)
        b["layout_type"] = layout_type
        b["is_noise"] = is_noise

    return blocks


def _compute_body_size(text_blocks: list[dict]) -> float:
    sizes = [b["font_size"] for b in text_blocks if b.get("font_size", 0) > 0]
    return median(sizes) if sizes else 12.0


def _compute_page_heights(blocks: list[dict]) -> dict[int, float]:
    heights: dict[int, float] = {}
    for b in blocks:
        bbox = b.get("bbox", (0, 0, 0, 0))
        page = b.get("page", 0)
        if page not in heights or bbox[3] > heights[page]:
            heights[page] = bbox[3]
    return heights


def _find_repeated_texts(
    text_blocks: list[dict], threshold: float = 0.6
) -> set[str]:
    page_counts: Counter[str] = Counter()
    pages_seen: dict[str, set[int]] = {}
    for b in text_blocks:
        text = b.get("text", "").strip()
        if not text:
            continue
        page = b.get("page", 0)
        if text not in pages_seen:
            pages_seen[text] = set()
        if page not in pages_seen[text]:
            pages_seen[text].add(page)
            page_counts[text] += 1

    total_pages = len({b.get("page", 0) for b in text_blocks})
    if total_pages == 0:
        return set()
    return {t for t, c in page_counts.items() if c / total_pages >= threshold}


def _feature_vector(
    block: dict,
    body_size: float,
    page_heights: dict[int, float],
    repeated_texts: set[str],
) -> dict[str, Any]:
    text = block.get("text", "").strip()
    font_size = block.get("font_size", body_size)
    bbox = block.get("bbox", (0, 0, 0, 0))
    page = block.get("page", 0)
    page_h = page_heights.get(page, 1.0)

    return {
        "text": text,
        "font_size": font_size,
        "font_size_ratio": font_size / body_size if body_size > 0 else 1.0,
        "is_all_caps": len(text) > 1 and text.isupper(),
        "is_short": len(text) < 20,
        "starts_with_number": bool(re.match(r"^\d", text)),
        "ends_with_punctuation": bool(re.search(r"[.!?:;]$", text)),
        "contains_page_number_pattern": bool(re.fullmatch(r"\d+", text)),
        "relative_y_position": bbox[1] / page_h if page_h > 0 else 0.5,
        "is_repeated_across_pages": text in repeated_texts,
    }


def _classify(
    fv: dict[str, Any], body_size: float, block: dict
) -> tuple[str, bool]:
    ratio = fv["font_size_ratio"]

    if ratio >= 1.6:
        return "heading", False
    if ratio >= 1.3:
        return "heading", False
    if ratio >= 1.15:
        return "heading", False

    if block.get("inside_table"):
        return "table_cell", False

    if _is_noise(fv, body_size):
        return _noise_type(fv), True

    best = max(_score_classes(fv), key=_score_classes(fv).get)
    return best, False


def _is_noise(fv: dict[str, Any], body_size: float) -> bool:
    if fv["is_repeated_across_pages"]:
        return True
    if (
        fv["contains_page_number_pattern"]
        and fv["is_short"]
        and fv["relative_y_position"] > 0.75
    ):
        return True
    return bool(fv["font_size"] < body_size * 0.6 and fv["is_short"])


def _noise_type(fv: dict[str, Any]) -> str:
    if fv["contains_page_number_pattern"]:
        return "page_number"
    if fv["relative_y_position"] < 0.15:
        return "header"
    if fv["relative_y_position"] > 0.85:
        return "footer"
    return "noise"


def _score_classes(fv: dict[str, Any]) -> dict[str, float]:
    r = fv["font_size_ratio"]
    caps = fv["is_all_caps"]
    short = fv["is_short"]
    starts_num = fv["starts_with_number"]
    ends_punct = fv["ends_with_punctuation"]
    y_pos = fv["relative_y_position"]

    scores: dict[str, float] = {}

    s = 0.0
    if 0.9 <= r < 1.15:
        s += 0.5
    if caps:
        s += 1.5
    if short:
        s += 0.3
    scores["heading"] = s

    s = 0.0
    if 0.85 <= r <= 1.15:
        s += 1.0
    if ends_punct:
        s += 1.0
    if short:
        s -= 1.5
    if caps:
        s -= 0.5
    scores["paragraph"] = s

    s = 0.0
    if short:
        s += 1.5
    if not ends_punct:
        s += 0.5
    if starts_num:
        s -= 0.5
    scores["short_text"] = s

    s = 0.0
    if y_pos < 0.15:
        s += 2.0
    if short:
        s += 0.5
    scores["header"] = s

    s = 0.0
    if y_pos > 0.85:
        s += 2.0
    if short:
        s += 0.5
    scores["footer"] = s

    s = 0.0
    if fv["contains_page_number_pattern"]:
        s += 3.0
    if short:
        s += 1.0
    scores["page_number"] = s

    s = 0.0
    if starts_num:
        s += 2.0
    if short:
        s += 0.5
    scores["list_item"] = s

    s = 0.0
    if short:
        s += 0.5
    if not starts_num:
        s += 0.3
    scores["table_cell"] = s

    return scores
