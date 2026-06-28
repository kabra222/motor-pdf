from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    import ot

    HAS_POT = True
except ImportError:
    HAS_POT = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class Segment:
    text: str
    start_char: int
    end_char: int
    depth: int = 0
    heading: Optional[str] = None
    page: int = 0


@dataclass
class SegmentationResult:
    segments: list[Segment] = field(default_factory=list)
    method: str = "heading"


_LEGAL_TERMS = {
    "capítulo", "artigo", "seção", "parágrafo", "inciso", "alínea",
    "lei", "decreto", "constituição", "emenda", "resolução",
    "contrato", "cláusula", "título", "livro", "parte",
    "subseção", "anexo", "considerando", "diante do exposto",
    "tese", "mérito", "pedido", "defesa", "recurso",
    "sentença", "acórdão", "decisão", "voto", "relatório",
}


def _extract_sentence_features(
    text: str, blocks: list[dict] | None = None
) -> tuple[list[str], np.ndarray | None]:
    lines = text.split("\n")
    units: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("##") or stripped.startswith("#"):
            units.append(stripped)
        elif len(stripped) > 20:
            units.append(stripped)
    if not units:
        units = [l.strip() for l in lines if len(l.strip()) > 20]
    if not units:
        return [], None
    if not HAS_NUMPY:
        return units, None

    feat = np.zeros((len(units), 80), dtype=np.float32)
    for i, unit in enumerate(units):
        words = unit.lower().split()
        total = max(len(words), 1)
        feat[i, 0] = len(unit) / 500.0
        feat[i, 1] = len(words) / 100.0
        feat[i, 2] = sum(1 for w in words if w in _LEGAL_TERMS) / max(total, 1)
        feat[i, 3] = sum(1 for c in unit if c.isdigit()) / max(len(unit), 1)
        feat[i, 4] = 1.0 if unit.startswith("#") else 0.0
        feat[i, 5] = 1.0 if (unit.startswith("##") or unit.startswith("# ")) else 0.0
        heads = {"capítulo", "título", "art.", "seção", "subseção", "livro", "parte"}
        feat[i, 6] = 1.0 if any(unit.lower().startswith(h) for h in heads) else 0.0
        feat[i, 7] = min(total, 100) / 100.0
        feat[i, 8] = sum(1 for c in unit if c in ".,;:!?") / max(len(unit), 1)
        feat[i, 9] = 1.0 if re.match(r"^[\d.]+[\s]", unit) else 0.0
        for j, w in enumerate(words[:25]):
            h = hash(w) % 68
            feat[i, 12 + h] += 1.0

    return units, feat


def segment_bcpd(
    text: str,
    blocks: list[dict] | None = None,
    max_segments: int = 100,
    penalty: float | None = None,
) -> SegmentationResult:
    if not HAS_RUPTURES:
        return SegmentationResult(segments=[], method="bcpd_unavailable")

    sentences, feat = _extract_sentence_features(text, blocks)
    if feat is None or len(sentences) < 5:
        return SegmentationResult(segments=[], method="bcpd_no_data")

    n = feat.shape[0]
    if n < 5:
        return SegmentationResult(segments=[], method="bcpd_too_short")

    algo = rpt.Pelt(model="l2", min_size=5, jump=1).fit(feat)
    pen = penalty or n * 0.02
    bkps = algo.predict(pen=pen)
    n_bkps = len(bkps)
    if n_bkps > max_segments:
        pen *= 2
        bkps = algo.predict(pen=pen)
    elif n_bkps < 3:
        pen *= 0.3
        bkps = algo.predict(pen=pen)

    boundaries = sorted(set([0] + [b for b in bkps if 0 < b < n] + [n]))

    segments = []
    char_pos = 0
    for i in range(len(boundaries) - 1):
        start_s = boundaries[i]
        end_s = boundaries[i + 1]
        seg_text = " ".join(sentences[start_s:end_s])
        if not seg_text.strip():
            continue
        seg_len = len(seg_text)
        from app.engine.chunker import _count_tokens

        tok = _count_tokens(seg_text)
        if tok < 10 and i > 0:
            prev = segments[-1]
            segments[-1] = Segment(
                text=prev.text + "\n" + seg_text,
                start_char=prev.start_char,
                end_char=char_pos + seg_len,
                depth=prev.depth,
                heading=prev.heading,
                page=prev.page,
            )
            char_pos += seg_len
            continue

        depth = _infer_depth(seg_text, blocks)
        heading = _infer_heading(seg_text)
        segments.append(
            Segment(
                text=seg_text,
                start_char=char_pos,
                end_char=char_pos + seg_len,
                depth=depth,
                heading=heading,
                page=0,
            )
        )
        char_pos += seg_len

    return SegmentationResult(segments=segments, method="bcpd_pelt")


def segment_headings(
    text: str, blocks: list[dict] | None = None
) -> SegmentationResult:
    segments = []
    if not blocks:
        return SegmentationResult(segments=[], method="heading_empty")

    sections: list[dict] = []
    current: dict = {"heading": None, "level": 0, "text_parts": [], "pages": set()}

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

    char_pos = 0
    for sec in sections:
        header = (
            f"{'#' * sec['level']} {sec['heading']}\n" if sec["heading"] else ""
        )
        body = "\n".join(sec["text_parts"])
        sec_text = (header + body).strip()
        if not sec_text:
            continue

        depth = sec.get("level", 0)
        segments.append(
            Segment(
                text=sec_text,
                start_char=char_pos,
                end_char=char_pos + len(sec_text),
                depth=depth,
                heading=sec["heading"],
                page=min(sec["pages"]) if sec["pages"] else 0,
            )
        )
        char_pos += len(sec_text)

    return SegmentationResult(segments=segments, method="heading")


def segment_hybrid(
    text: str,
    blocks: list[dict] | None = None,
    chunk_size: int = 2000,
) -> SegmentationResult:
    heading_result = segment_headings(text, blocks)
    if heading_result.segments:
        return heading_result

    if HAS_RUPTURES:
        bcpd_result = segment_bcpd(text, blocks)
        if bcpd_result.segments:
            return bcpd_result

    return SegmentationResult(segments=[], method="none")


def _infer_depth(seg_text: str, blocks: list[dict] | None = None) -> int:
    if not blocks:
        return 0
    lines = seg_text.split("\n")
    first_line = lines[0].strip() if lines else ""

    for block in blocks:
        if block.get("type") != "text":
            continue
        if block.get("is_heading") and block.get("text", "").strip() in first_line:
            return block.get("heading_level", 1)

    heading_match = re.match(r"^(#{1,6})\s", seg_text)
    if heading_match:
        return len(heading_match.group(1))

    if re.match(r"^(CAPÍTULO|TÍTULO|LIVRO|PARTE|SEÇÃO)\s", seg_text.upper()):
        return 1
    if re.match(r"^\d+\.\d+\.\s", seg_text):
        return 2
    if re.match(r"^\d+\.\s", seg_text):
        return 1
    if re.match(r"^[A-Z][a-zá-ú]+(\s[A-Z][a-zá-ú]+){0,3}$", seg_text[:60]):
        return 1
    return 0


def _infer_heading(seg_text: str) -> Optional[str]:
    lines = [l.strip() for l in seg_text.split("\n") if l.strip()]
    if not lines:
        return None
    first = lines[0]
    heading_match = re.match(r"^#{1,6}\s+(.+)$", first)
    if heading_match:
        return heading_match.group(1).strip()
    if len(first) < 120 and not first.endswith("."):
        return first
    return None


def apply_guardrail(
    segments: list[Segment], text: str
) -> list[Segment]:
    return segments
