from __future__ import annotations

from collections import defaultdict

GAP_THRESHOLD = 15.0


def detect_column_gaps(blocks: list[dict]) -> list[float]:
    text_blocks = [b for b in blocks if b.get("type") == "text" and b.get("bbox")]
    if len(text_blocks) < 3:
        return []

    sorted_h = sorted(text_blocks, key=lambda b: b["bbox"][0])
    gaps: list[float] = []
    for i in range(len(sorted_h) - 1):
        gap = sorted_h[i + 1]["bbox"][0] - sorted_h[i]["bbox"][2]
        if gap > GAP_THRESHOLD:
            gaps.append(gap)
    if not gaps:
        return []

    median_gap = sorted(gaps)[len(gaps) // 2]
    column_boundaries: set[float] = set()
    for i in range(len(sorted_h) - 1):
        gap = sorted_h[i + 1]["bbox"][0] - sorted_h[i]["bbox"][2]
        if abs(gap - median_gap) < median_gap * 0.5:
            column_boundaries.add(
                (sorted_h[i]["bbox"][2] + sorted_h[i + 1]["bbox"][0]) / 2
            )

    return sorted(column_boundaries)


def assign_columns(
    blocks: list[dict], boundaries: list[float]
) -> dict[int, list[dict]]:
    cols: dict[int, list[dict]] = defaultdict(list)
    for b in blocks:
        if b.get("type") != "text" or not b.get("bbox"):
            continue
        cx = (b["bbox"][0] + b["bbox"][2]) / 2
        assigned = False
        for ci, bound in enumerate(boundaries):
            if cx < bound:
                cols[ci].append(b)
                assigned = True
                break
        if not assigned:
            cols[len(boundaries)].append(b)
    return cols


def reorder_blocks(
    blocks: list[dict], boundaries: list[float] | None = None
) -> list[dict]:
    if boundaries is None:
        boundaries = detect_column_gaps(blocks)
    if not boundaries:
        return blocks

    text_blocks = [b for b in blocks if b.get("type") == "text"]
    non_text = [b for b in blocks if b.get("type") != "text"]

    cols = assign_columns(text_blocks, boundaries)
    reordered: list[dict] = []
    for col_idx in range(len(boundaries) + 1):
        col_blocks = cols.get(col_idx, [])
        col_blocks.sort(key=lambda b: b["bbox"][1])
        reordered.extend(col_blocks)

    return reordered + non_text
