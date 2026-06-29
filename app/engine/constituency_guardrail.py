from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

HAS_STANZA = False
try:
    import stanza
    HAS_STANZA = True
except ImportError:
    pass


class ConstituencyGuardrail:
    def __init__(self, lang: str = "pt"):
        self.lang = lang
        self._nlp = None
        self._available = HAS_STANZA

    @property
    def is_available(self) -> bool:
        return self._available

    def _ensure_model(self):
        if self._nlp is not None:
            return
        if not HAS_STANZA:
            self._available = False
            return
        try:
            stanza.download(self.lang, verbose=False, logging_level="WARNING")
            self._nlp = stanza.Pipeline(
                self.lang,
                processors="tokenize,pos,constituency",
                use_gpu=False,
                verbose=False,
                logging_level="WARNING",
            )
        except Exception as exc:
            logger.debug("Stanza init failed: %s", exc)
            self._available = False

    def get_constituency_tree(self, text: str) -> list[dict]:
        if not self._available or not text.strip():
            return []
        self._ensure_model()
        if self._nlp is None:
            return []
        try:
            doc = self._nlp(text[:5000])
            trees = []
            for sent in doc.sentences:
                if sent.constituency:
                    trees.append({
                        "text": sent.text,
                        "tree": str(sent.constituency),
                        "depth": _tree_depth(sent.constituency),
                    })
            return trees
        except Exception as exc:
            logger.debug("Constituency parse failed: %s", exc)
            return []

    def find_constituent_boundaries(
        self, text: str, proposed_break: int
    ) -> list[int]:
        if not self._available or not text.strip():
            return [proposed_break]
        self._ensure_model()
        if self._nlp is None:
            return [proposed_break]
        try:
            doc = self._nlp(text[:8000])
            invalid_breaks = []
            char_pos = 0
            for sent in doc.sentences:
                sent_start = char_pos
                sent_end = sent_start + len(sent.text)
                if sent_start < proposed_break < sent_end:
                    invalid_breaks.append(proposed_break)
                    boundary = _find_nearest_safe_break(sent, text, proposed_break)
                    if boundary is not None:
                        invalid_breaks.append(boundary)
                char_pos = sent_end + 1
            if invalid_breaks:
                return [b for b in invalid_breaks if b != proposed_break] or [proposed_break]
            return [proposed_break]
        except Exception as exc:
            logger.debug("Constituency boundary find failed: %s", exc)
            return [proposed_break]

    def is_safe_break(self, text: str, char_pos: int) -> bool:
        if not self._available or not text.strip():
            return True
        self._ensure_model()
        if self._nlp is None:
            return True
        try:
            window = text[max(0, char_pos - 200):char_pos + 200]
            offset = min(char_pos, 200)
            doc = self._nlp(window[:5000])
            for sent in doc.sentences:
                start_in_window = text.find(sent.text, max(0, char_pos - 200))
                if start_in_window == -1:
                    continue
                sent_start = start_in_window
                sent_end = sent_start + len(sent.text)
                if sent_start < char_pos < sent_end:
                    invalids = _find_breakpoints_in_sentence(sent.constituency, sent.text)
                    rel_char = char_pos - sent_start
                    for inv in invalids:
                        if abs(inv - rel_char) < 5:
                            return False
            return True
        except Exception:
            return True


def _tree_depth(tree) -> int:
    s = str(tree)
    max_depth = 0
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ")":
            depth -= 1
    return max_depth


def _find_nearest_safe_break(
    sent, text: str, proposed_break: int
) -> int | None:
    if not sent.constituency:
        return None
    boundaries = _find_breakpoints_in_sentence(sent.constituency, sent.text)
    if not boundaries:
        return None
    sent_start = text.find(sent.text)
    if sent_start == -1:
        return None
    rel_proposed = proposed_break - sent_start
    for b in sorted(boundaries):
        if b > rel_proposed:
            return sent_start + b
    return sent_start + len(sent.text)


def _find_breakpoints_in_sentence(tree, sent_text: str) -> list[int]:
    s = str(tree)
    breakpoints = []
    depth = 0
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth <= 1 and c in (".", "!", "?"):
            char_in_sent = len(s[:i].split(")")[-1])
            if 5 < char_in_sent < len(sent_text) - 5:
                breakpoints.append(char_in_sent)
    return sorted(set(breakpoints)) if breakpoints else [len(sent_text)]


def safe_break_positions(text: str, proposed: list[int]) -> list[int]:
    guardrail = ConstituencyGuardrail()
    safe = []
    for pos in proposed:
        if guardrail.is_safe_break(text, pos):
            safe.append(pos)
        else:
            boundaries = guardrail.find_constituent_boundaries(text, pos)
            safe.extend(b for b in boundaries if b != pos)
    return sorted(set(safe))
