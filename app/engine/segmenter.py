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

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


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


def _compute_embedding_features(
    units: list[str],
) -> np.ndarray | None:
    if not HAS_SENTENCE_TRANSFORMERS or len(units) < 3:
        return None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(units, show_progress_bar=False)
        if len(embs) < 3:
            return None
        return np.array(embs, dtype=np.float32)
    except Exception:
        return None


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


if HAS_RUPTURES and HAS_POT:

    class WassersteinTopologicalCost(rpt.base.BaseCost):
        model = "custom_wasserstein_1d"
        min_size = 3

        def __init__(self):
            self.signal = None
            self.n_samples = None

        def fit(self, signal):
            self.signal = signal
            self.n_samples = signal.shape[0]
            return self

        def error(self, start, end):
            if end - start < self.min_size * 2:
                return 0.0
            sub = self.signal[start:end]
            mid = len(sub) // 2
            mag_p = np.linalg.norm(sub[:mid], axis=1)
            mag_q = np.linalg.norm(sub[mid:], axis=1)
            sp = np.sum(mag_p)
            sq = np.sum(mag_q)
            p_dist = mag_p / sp if sp != 0 else mag_p
            q_dist = mag_q / sq if sq != 0 else mag_q
            cp = np.arange(len(p_dist), dtype=np.float64)
            cq = np.arange(len(q_dist), dtype=np.float64)
            w1 = ot.wasserstein_1d(cp, cq, p_dist, q_dist, p=1)
            return float(w1 * (end - start))

else:

    class WassersteinTopologicalCost:
        pass


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

    embed_feat: np.ndarray | None = None
    embed_method: str = ""
    if use_embeddings:
        embed_feat = _compute_embedding_features(sentences)
        if embed_feat is not None and embed_feat.shape[0] == n:
            embed_feat = (embed_feat - embed_feat.mean(axis=0)) / (
                embed_feat.std(axis=0) + 1e-8
            )
            feat = np.concatenate([feat, embed_feat], axis=1)
            embed_method = "_simcse"

    if HAS_POT:
        custom = WassersteinTopologicalCost()
        algo = rpt.Pelt(custom_cost=custom, min_size=5, jump=1).fit(feat)
    else:
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

    base = "bcpd_pelt_wasserstein" if HAS_POT else "bcpd_pelt"
    return SegmentationResult(segments=segments, method=base + embed_method)


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
    use_embeddings: bool = False,
) -> SegmentationResult:
    heading_result = segment_headings(text, blocks)
    if heading_result.segments:
        heading_result.segments = apply_guardrail(heading_result.segments, text)
        return heading_result

    if HAS_RUPTURES:
        bcpd_result = segment_bcpd(text, blocks, use_embeddings=use_embeddings)
        if bcpd_result.segments:
            bcpd_result.segments = apply_guardrail(bcpd_result.segments, text)
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


_SENTENCE_END = re.compile(r"[.!?][\"']?\s*$")

_ANAPHORA_PATTERNS = re.compile(
    r"^\s*(O|A|Os|As|Este|Esta|Estes|Estas|Esse|Essa|Esses|Essas|"
    r"Aquele|Aquela|Aqueles|Aquelas|Seu|Sua|Seus|Suas|Lhe|Lhes|"
    r"O referido|A referida|O mencionado|A mencionada|O citado|A citada|"
    r"Tal|Tais|Mesmo|Mesma|Mesmos|Mesmas|"
    r"O presente|A presente|O sobredito|A sobredita)\b",
    re.IGNORECASE,
)

_SHORT_CONJUNCTIONS = re.compile(
    r"^\s*(e|mas|porém|contudo|todavia|entretanto|no entanto|"
    r"pois|portanto|logo|assim|desse modo|"
    r"que|o qual|a qual|os quais|as quais|"
    r"como|conforme|segundo|"
    r"além disso|ademais|outrossim)\b",
    re.IGNORECASE,
)

_MIN_GUARDRAIL_TOKENS = 30


def _count_tokens_simple(text: str) -> int:
    return len(text.split())


def _check_orphan_start(segments: list[Segment], idx: int) -> bool:
    if idx <= 0 or idx >= len(segments):
        return False
    cur = segments[idx].text.lstrip()
    prev = segments[idx - 1].text
    if _ANAPHORA_PATTERNS.match(cur) and _SENTENCE_END.search(prev) is None:
        return True
    if _SHORT_CONJUNCTIONS.match(cur) and _SENTENCE_END.search(prev) is None:
        return True
    return False


def apply_guardrail(
    segments: list[Segment], text: str
) -> list[Segment]:
    if not segments:
        return segments
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        cur_tok = _count_tokens_simple(seg.text)
        prev_tok = _count_tokens_simple(prev.text)
        if _check_orphan_start(merged, len(merged)):
            prev.text = prev.text.rstrip() + "\n" + seg.text
            prev.end_char = seg.end_char
            if seg.depth > prev.depth:
                prev.depth = seg.depth
            if seg.heading and not prev.heading:
                prev.heading = seg.heading
            continue
        if cur_tok < _MIN_GUARDRAIL_TOKENS and prev_tok > _MIN_GUARDRAIL_TOKENS * 3:
            prev.text = prev.text.rstrip() + "\n" + seg.text
            prev.end_char = seg.end_char
            if seg.depth > prev.depth:
                prev.depth = seg.depth
            if seg.heading and not prev.heading:
                prev.heading = seg.heading
            continue
        merged.append(seg)
    return merged
