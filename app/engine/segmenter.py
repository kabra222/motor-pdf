from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.engine.constituency_guardrail import ConstituencyGuardrail
from app.engine.coreference import CoreferenceTracker
from app.engine.simcse_loss import compute_simcse_embeddings, estimate_anisotropy, info_nce_loss
from app.engine.wasserstein_cost import WassersteinTopologicalCost

logger = logging.getLogger(__name__)

HAS_RUPTURES = False
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    pass

HAS_POT = False
try:
    import ot
    HAS_POT = True
except ImportError:
    pass

HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass


@dataclass
class Segment:
    text: str
    start_char: int
    end_char: int
    depth: int = 0
    heading: str | None = None
    page: int = 0
    confidence: float = 1.0
    class_: str = "paragraph"


@dataclass
class SegmentationResult:
    segments: list[Segment] = field(default_factory=list)
    method: str = "heading"
    anisotropy: dict | None = None
    coreference_density: float = 0.0


_LEGAL_TERMS = {
    "capítulo", "artigo", "seção", "parágrafo", "inciso", "alínea",
    "lei", "decreto", "constituição", "emenda", "resolução",
    "contrato", "cláusula", "título", "livro", "parte",
    "subseção", "anexo", "considerando", "diante do exposto",
    "tese", "mérito", "pedido", "defesa", "recurso",
    "sentença", "acórdão", "decisão", "voto", "relatório",
}


def _compute_embedding_features(units: list[str]) -> np.ndarray | None:
    if not HAS_SENTENCE_TRANSFORMERS or len(units) < 3:
        return None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        embs = model.encode(units, show_progress_bar=False)
        if len(embs) < 3:
            return None
        return np.array(embs, dtype=np.float32)
    except Exception:
        return None


def _extract_sentence_features(text: str, blocks: list[dict] | None = None) -> tuple[list[str], np.ndarray | None]:
    lines = text.split("\n")
    units: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("##") or stripped.startswith("#") or len(stripped) > 20:
            units.append(stripped)
    if not units:
        units = [l.strip() for l in lines if len(l.strip()) > 20]
    if not units:
        return [], None

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
        for _j, w in enumerate(words[:25]):
            h = hash(w) % 68
            feat[i, 12 + h] += 1.0

    return units, feat


def segment_bcpd(
    text: str,
    blocks: list[dict] | None = None,
    max_segments: int = 100,
    penalty: float | None = None,
    use_embeddings: bool = False,
) -> SegmentationResult:
    if not HAS_RUPTURES:
        return SegmentationResult(segments=[], method="bcpd_unavailable")

    sentences, feat = _extract_sentence_features(text, blocks)
    if feat is None or len(sentences) < 5:
        return SegmentationResult(segments=[], method="bcpd_no_data")

    n = feat.shape[0]
    if n < 5:
        return SegmentationResult(segments=[], method="bcpd_too_short")

    anisotropy = None
    embed_method = ""

    if use_embeddings:
        simcse_embs = compute_simcse_embeddings(sentences, dropout_mask=0.1)
        if simcse_embs is not None and simcse_embs.shape[0] >= n:
            aligned = simcse_embs[:n]
            aligned = (aligned - aligned.mean(axis=0)) / (aligned.std(axis=0) + 1e-8)
            feat = np.concatenate([feat, aligned], axis=1)
            anisotropy = estimate_anisotropy(aligned)
            info_nce = info_nce_loss(aligned, temperature=0.05)
            if anisotropy and info_nce > 0:
                anisotropy["info_nce_loss"] = round(info_nce, 4)
            embed_method = "_simcse"

    use_wasserstein = WassersteinTopologicalCost.is_available()
    if use_wasserstein:
        custom = WassersteinTopologicalCost.create()
        if custom is not None:
            algo = rpt.Pelt(custom_cost=custom, min_size=5, jump=1).fit(feat)
        else:
            algo = rpt.Pelt(model="l2", min_size=5, jump=1).fit(feat)
    else:
        algo = rpt.Pelt(model="l2", min_size=5, jump=1).fit(feat)

    pen = penalty or n * 0.02
    bkps = algo.predict(pen=pen)
    n_bkps = len(bkps)

    if n_bkps > max_segments:
        pen *= 2
        bkps = algo.predict(pen=pen)
    elif n_bkps < 3 and penalty is None:
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
                confidence=_estimate_confidence(boundaries, i, n, feat),
            )
        )
        char_pos += seg_len

    segments = _apply_constituency_guardrail(segments, text)
    segments = _apply_coreference_guardrail(segments)

    base = "bcpd_pelt_wasserstein" if use_wasserstein else "bcpd_pelt"
    return SegmentationResult(
        segments=segments,
        method=base + embed_method,
        anisotropy=anisotropy,
    )


def _estimate_confidence(
    boundaries: list[int], idx: int, n: int, feat: np.ndarray
) -> float:
    if idx <= 0 or idx >= len(boundaries) - 2:
        return 1.0
    start = boundaries[idx]
    end = boundaries[idx + 1]
    if start < 0 or end > n or end - start < 1:
        return 1.0
    try:
        if start < feat.shape[0] - 1 and end < feat.shape[0]:
            sub = feat[start:end]
            if sub.shape[0] >= 2:
                var = float(np.mean(np.var(sub, axis=0)))
                return min(1.0, max(0.1, 1.0 / (1.0 + var)))
    except Exception:
        pass
    return 0.8


def _apply_constituency_guardrail(
    segments: list[Segment], text: str
) -> list[Segment]:
    if not segments:
        return segments
    guardrail = ConstituencyGuardrail()
    if not guardrail.is_available:
        return segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        proposed = prev.end_char
        if not guardrail.is_safe_break(text, proposed):
            prev.text = prev.text.rstrip() + "\n\n" + seg.text
            prev.end_char = seg.end_char
            if seg.depth > prev.depth:
                prev.depth = seg.depth
            if seg.heading and not prev.heading:
                prev.heading = seg.heading
            continue
        merged.append(seg)
    return merged


def _apply_coreference_guardrail(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return segments
    tracker = CoreferenceTracker()
    texts = [s.text for s in segments]
    orphan_risk = tracker.detect_orphan_risk(texts)
    if not orphan_risk:
        return segments
    merged = []
    for i, seg in enumerate(segments):
        if not merged:
            merged.append(seg)
            continue
        if i in orphan_risk:
            prev = merged[-1]
            prev.text = prev.text.rstrip() + "\n\n" + seg.text
            prev.end_char = seg.end_char
            if seg.depth > prev.depth:
                prev.depth = seg.depth
            if seg.heading and not prev.heading:
                prev.heading = seg.heading
        else:
            merged.append(seg)
    return merged


def segment_headings(text: str, blocks: list[dict] | None = None) -> SegmentationResult:
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

    segments = _apply_coreference_guardrail(segments)
    return SegmentationResult(segments=segments, method="heading")


def segment_hybrid(
    text: str,
    blocks: list[dict] | None = None,
    chunk_size: int = 2000,
    use_embeddings: bool = False,
) -> SegmentationResult:
    heading_result = segment_headings(text, blocks)
    if heading_result.segments:
        heading_result.segments = _apply_merge_guardrail(heading_result.segments, text)
        return heading_result

    if HAS_RUPTURES:
        bcpd_result = segment_bcpd(text, blocks, use_embeddings=use_embeddings)
        if bcpd_result.segments:
            bcpd_result.segments = _apply_merge_guardrail(bcpd_result.segments, text)
            return bcpd_result

    return SegmentationResult(segments=[], method="none")


def _apply_merge_guardrail(segments: list[Segment], text: str) -> list[Segment]:
    segments = _apply_constituency_guardrail(segments, text)
    segments = _apply_coreference_guardrail(segments)
    return _apply_min_size_guardrail(segments)


def _apply_min_size_guardrail(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if len(seg.text.split()) < 10 and len(prev.text.split()) > 30:
            prev.text = prev.text.rstrip() + "\n\n" + seg.text
            prev.end_char = seg.end_char
            if seg.depth > prev.depth:
                prev.depth = seg.depth
            if seg.heading and not prev.heading:
                prev.heading = seg.heading
        else:
            merged.append(seg)
    return merged


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


def _infer_heading(seg_text: str) -> str | None:
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
