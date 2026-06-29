from __future__ import annotations

import re

_UNICODE_SUPER = str.maketrans({
    "²": "^2",
    "³": "^3",
    "¹": "^1",
    "⁰": "^0",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
})

_UNICODE_SUB = str.maketrans({
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
})

_SUBSCRIPT_CARET = re.compile(r"\^\{([^}]+)\}")
_SUPERSCRIPT_CARET = re.compile(r"(?<=[a-zA-Z)\]])\^(\d+)")
_SUBSCRIPT_DIGIT = re.compile(r"(?<=[a-zA-Z)])(\d{2,})(?=[a-zA-Z\s,;.]|$)")

_GREEK_ENCODING = re.compile(r"(?<![a-zA-Z])([abgdezhqiklmnxoprstyfcyw])(?![a-zA-Z])")


def normalize_superscript_unicode(text: str) -> str:
    return text.translate(_UNICODE_SUPER)


def normalize_subscript_unicode(text: str) -> str:
    return text.translate(_UNICODE_SUB)


def fix_caret_notation(text: str) -> str:
    text = _SUBSCRIPT_CARET.sub(r"_{\1}", text)
    text = _SUPERSCRIPT_CARET.sub(r"^{\1}", text)
    return text


def mark_digit_subscripts(text: str) -> str:
    return _SUBSCRIPT_DIGIT.sub(r"_{\1}", text)


def has_math_patterns(text: str) -> bool:
    if re.search(r"[\^_²³¹⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉{}]", text):
        return True
    if re.search(r"(?<=[a-zA-Z)])\d{2,}(?=[a-zA-Z\s,;.]|$)", text):
        return True
    return False


def reconstruct_math_text(text: str) -> str:
    text = normalize_superscript_unicode(text)
    text = normalize_subscript_unicode(text)
    text = fix_caret_notation(text)
    text = mark_digit_subscripts(text)
    return text
