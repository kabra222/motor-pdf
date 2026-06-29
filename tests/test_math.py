from __future__ import annotations

from app.engine.math import (
    fix_caret_notation,
    has_math_patterns,
    mark_digit_subscripts,
    normalize_subscript_unicode,
    normalize_superscript_unicode,
    reconstruct_math_text,
)


class TestNormalizeSuperscript:
    def test_superscript_unicode(self):
        assert normalize_superscript_unicode("x²") == "x^2"
        assert normalize_superscript_unicode("x³") == "x^3"
        assert normalize_superscript_unicode("x¹") == "x^1"
        assert normalize_superscript_unicode("x⁰") == "x^0"
        assert normalize_superscript_unicode("y⁴⁵") == "y^4^5"

    def test_plain_text_unchanged(self):
        assert normalize_superscript_unicode("Texto normal") == "Texto normal"


class TestNormalizeSubscript:
    def test_subscript_unicode(self):
        assert normalize_subscript_unicode("x₂") == "x_2"
        assert normalize_subscript_unicode("x₃") == "x_3"
        assert normalize_subscript_unicode("a₁") == "a_1"

    def test_plain_text_unchanged(self):
        assert normalize_subscript_unicode("Texto normal") == "Texto normal"


class TestFixCaretNotation:
    def test_subscript_caret(self):
        assert fix_caret_notation("x^{2}") == "x_{2}"
        assert fix_caret_notation("x_{i}") == "x_{i}"
        assert fix_caret_notation("x^{n+1}") == "x_{n+1}"

    def test_superscript_caret(self):
        assert fix_caret_notation("x^2") == "x^{2}"
        assert fix_caret_notation("x^3 + y^2") == "x^{3} + y^{2}"
        assert fix_caret_notation("x^n") == "x^n"
        assert fix_caret_notation("x^(n+1)") == "x^(n+1)"


class TestMarkDigitSubscripts:
    def test_two_digit_subscript(self):
        assert mark_digit_subscripts("x23") == "x_{23}"
        assert mark_digit_subscripts("x23 + y45") == "x_{23} + y_{45}"

    def test_within_text(self):
        assert mark_digit_subscripts("variable12 = 5") == "variable_{12} = 5"

    def test_no_match_single_digit(self):
        assert mark_digit_subscripts("x2") == "x2"

    def test_no_match_plain_text(self):
        assert mark_digit_subscripts("texto normal") == "texto normal"


class TestHasMathPatterns:
    def test_superscript_unicode(self):
        assert has_math_patterns("x² + y²")

    def test_subscript_unicode(self):
        assert has_math_patterns("x₂ + y₂")

    def test_caret_notation(self):
        assert has_math_patterns("x^{2}")
        assert has_math_patterns("x_{i}")

    def test_digit_subscript(self):
        assert has_math_patterns("x23")

    def test_no_pattern(self):
        assert not has_math_patterns("texto normal")
        assert not has_math_patterns("apenas palavras comuns")


class TestReconstructMathText:
    def test_superscript_conversion(self):
        result = reconstruct_math_text("x² + y³")
        assert "x^{2}" in result
        assert "y^{3}" in result

    def test_subscript_conversion(self):
        result = reconstruct_math_text("x₂ + y₃")
        assert "x_2" in result or "x2" in result

    def test_caret_to_brace(self):
        result = reconstruct_math_text("x^{n+1}")
        assert "x_{" in result

    def test_complex_expression(self):
        result = reconstruct_math_text("E = mc²")
        assert "mc^{2}" in result

    def test_digit_subscript(self):
        result = reconstruct_math_text("x23")
        assert "x_{23}" in result

    def test_plain_text_unchanged(self):
        assert reconstruct_math_text("Ola mundo") == "Ola mundo"
