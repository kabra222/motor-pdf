from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

HAS_SPACY = False
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    pass

HAS_STANZA = False
try:
    import stanza
    HAS_STANZA = True
except ImportError:
    pass

_PRONOMES_ANA_FORICOS = re.compile(
    r"\b(Este|Esta|Estes|Estas|Esse|Essa|Esses|Essas|"
    r"Aquele|Aquela|Aqueles|Aquelas|"
    r"Seu|Sua|Seus|Suas|Lhe|Lhes|"
    r"O|A|Os|As|"
    r"Mesmo|Mesma|Mesmos|Mesmas|"
    r"Tal|Tais|"
    r"O referido|A referida|O mencionado|A mencionada|"
    r"O citado|A citada|O sobredito|A sobredita|"
    r"O presente|A presente|"
    r"cujo|cuja|cujos|cujas)\b",
    re.IGNORECASE,
)

_REFERENCIA_PROCESSAL = re.compile(
    r"\b(Apelante|Apelado|Recorrente|Recorrido|"
    r"Reclamante|Reclamado|Autor|Réu|Requirente|Requerido|"
    r"Embargante|Embargado|Exequente|Executado|"
    r"Impetrante|Impetrado|Agravante|Agravado)\b",
    re.IGNORECASE,
)

_PRONOME_TRATAMENTO = re.compile(
    r"\b(Vossa Excelência|Vossa Senhoria|Excelentíssimo|"
    r"Meritíssimo|Douto|Douta|DD\.)\b",
    re.IGNORECASE,
)

_CONJUNCAO_RETROSPECTIVA = re.compile(
    r"^\s*(Diante do exposto|Ante o exposto|Isto posto|"
    r"Nesse sentido|Dessa forma|Desse modo|"
    r"Assim sendo|Portanto|Contudo|Todavia|"
    r"No entanto|Entretanto|Por fim|Ademais)\b",
    re.IGNORECASE,
)


class CoreferenceTracker:
    def __init__(self, lang: str = "pt"):
        self.lang = lang
        self._nlp = None
        self._available = HAS_STANZA or HAS_SPACY

    @property
    def is_available(self) -> bool:
        return self._available

    def _ensure_model(self):
        if self._nlp is not None:
            return
        if HAS_STANZA:
            try:
                stanza.download(self.lang, verbose=False, logging_level="WARNING")
                self._nlp = stanza.Pipeline(
                    self.lang,
                    processors="tokenize,pos,ner,depparse",
                    use_gpu=False,
                    verbose=False,
                    logging_level="WARNING",
                )
            except Exception as exc:
                logger.debug("Stanza coref model failed: %s", exc)
                self._available = False
        elif HAS_SPACY:
            try:
                self._nlp = spacy.load(f"{self.lang}_core_news_sm")
            except Exception:
                try:
                    spacy.cli.download(f"{self.lang}_core_news_sm")
                    self._nlp = spacy.load(f"{self.lang}_core_news_sm")
                except Exception as exc:
                    logger.debug("spaCy coref model failed: %s", exc)
                    self._available = False

    def find_anaphora(self, segments: list[str]) -> list[dict]:
        anaphora = []
        for i, seg in enumerate(segments):
            ana_matches = _PRONOMES_ANA_FORICOS.findall(seg)
            proc_matches = _REFERENCIA_PROCESSAL.findall(seg)
            conj_matches = _CONJUNCAO_RETROSPECTIVA.findall(seg.split("\n")[0])
            if ana_matches or proc_matches or conj_matches:
                anaphora.append({
                    "segment_idx": i,
                    "pronomes": list(set(ana_matches)),
                    "referencias_processuais": list(set(proc_matches)),
                    "conjuncoes_retro": list(set(conj_matches)),
                    "has_anaphora": bool(ana_matches or conj_matches),
                    "has_reference": bool(proc_matches),
                })
        return anaphora

    def detect_orphan_risk(self, segments: list[str]) -> list[int]:
        anaphora = self.find_anaphora(segments)
        orphan_risk = []
        for entry in anaphora:
            idx = entry["segment_idx"]
            if idx == 0:
                continue
            if entry["has_anaphora"]:
                prev_seg = segments[idx - 1][:200]
                antecedente = self._find_antecedent(
                    segments[idx], prev_seg
                )
                if not antecedente:
                    orphan_risk.append(idx)
        return orphan_risk

    def _find_antecedent(self, current: str, previous: str) -> str | None:
        if not self._available:
            return None
        self._ensure_model()
        if self._nlp is None:
            return None
        try:
            pronomes = _PRONOMES_ANA_FORICOS.findall(current)
            if not pronomes:
                return _CONJUNCAO_RETROSPECTIVA.search(current).group(0) if _CONJUNCAO_RETROSPECTIVA.search(current) else None
            if HAS_STANZA:
                doc = self._nlp(previous[:2000])
                for ent in doc.entities:
                    if ent.type in ("PER", "ORG", "LAW"):
                        return ent.text
                nouns = [w.text for w in doc.sentences[0].words if w.upos in ("PROPN", "NOUN")]
                if nouns:
                    return nouns[0]
            elif HAS_SPACY:
                doc = self._nlp(previous[:2000])
                for ent in doc.ents:
                    if ent.label_ in ("PER", "ORG", "LAW", "MISC"):
                        return ent.text
                nouns = [token.text for token in doc if token.pos_ in ("PROPN", "NOUN")]
                if nouns:
                    return nouns[0]
            return None
        except Exception:
            return None

    def resolve_segments(self, segments: list[str]) -> list[str]:
        if len(segments) < 2:
            return segments
        resolved = [segments[0]]
        for i in range(1, len(segments)):
            orphan_risk = self.detect_orphan_risk(segments[:i + 1])
            if i in orphan_risk:
                resolved[-1] = resolved[-1].rstrip() + "\n\n" + segments[i]
            else:
                resolved.append(segments[i])
        return resolved

    def estimate_coreference_density(self, text: str) -> dict:
        sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
        if len(sentences) < 2:
            return {"density": 0.0, "total_anaphora": 0}
        total_ana = 0
        for sent in sentences:
            total_ana += len(_PRONOMES_ANA_FORICOS.findall(sent))
            total_ana += len(_REFERENCIA_PROCESSAL.findall(sent))
        return {
            "density": round(total_ana / max(len(sentences), 1), 4),
            "total_anaphora": total_ana,
        }
