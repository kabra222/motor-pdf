from __future__ import annotations

import re

from app.engine.classifier import (
    _COMMON_HEADING_WORDS,
    _NUMBERED_HEADING,
    _SHORT_UPPER_HEADING,
    _infer_heading_level,
    classify_blocks_builtin,
)


class TestNumberedHeading:
    def test_decimal_heading(self):
        assert _NUMBERED_HEADING.match("1.2.3 Section Title")
        assert _NUMBERED_HEADING.match("9.8 How poor countries get stuck")

    def test_standalone_number(self):
        assert _NUMBERED_HEADING.match("1 Introduction")
        assert _NUMBERED_HEADING.match("2 Background")
        assert _NUMBERED_HEADING.match("3 Model Architecture")

    def test_roman_numeral(self):
        assert _NUMBERED_HEADING.match("I. Introduction")
        assert _NUMBERED_HEADING.match("II. Literature Review")

    def test_single_letter(self):
        assert _NUMBERED_HEADING.match("A. Background")
        assert _NUMBERED_HEADING.match("B. Methods")

    def test_exercise_pattern(self):
        assert _NUMBERED_HEADING.match("EXERCISE 9.10")
        assert _NUMBERED_HEADING.match("Exercício 9.10")
        assert _NUMBERED_HEADING.match("EXERCISE 1")
        assert _NUMBERED_HEADING.match("PROBLEM 3.2")
        assert _NUMBERED_HEADING.match("QUESTÃO 5")
        assert _NUMBERED_HEADING.match("SECTION 2.1")
        assert _NUMBERED_HEADING.match("CHAPTER 4")
        assert _NUMBERED_HEADING.match("APPENDIX A")
        assert _NUMBERED_HEADING.match("ANEXO I")

    def test_rejects_table_data(self):
        assert not _NUMBERED_HEADING.match("0.0 5.77 24.6 0.2 4.95")
        assert not _NUMBERED_HEADING.match("3.14 2.71 1.41")

    def test_not_heading(self):
        assert not _NUMBERED_HEADING.match("texto normal")
        assert not _NUMBERED_HEADING.match("um parágrafo qualquer")


class TestHeadingLevelInference:
    def test_standalone_number_is_l1(self):
        assert _infer_heading_level("1 Introduction") == 1
        assert _infer_heading_level("2 Background") == 1

    def test_single_dot_is_l2(self):
        assert _infer_heading_level("9.8 How poor countries") == 2
        assert _infer_heading_level("3.1 Encoder and Decoder") == 2

    def test_double_dot_is_l3(self):
        assert _infer_heading_level("3.2.1 Scaled Dot-Product") == 3
        assert _infer_heading_level("1.2.3 Detail") == 3

    def test_roman_is_l1(self):
        assert _infer_heading_level("I. Introduction") == 1

    def test_letter_is_l2(self):
        assert _infer_heading_level("A. Background") == 2

    def test_keyword_is_l1(self):
        assert _infer_heading_level("EXERCISE 9.10") == 1
        assert _infer_heading_level("CHAPTER 4") == 1
        assert _infer_heading_level("APPENDIX A") == 1

    def test_plain_text_is_none(self):
        assert _infer_heading_level("texto normal") is None


class TestCommonHeadingWords:
    def test_abstract(self):
        assert "abstract" in _COMMON_HEADING_WORDS

    def test_conclusion(self):
        assert "conclusion" in _COMMON_HEADING_WORDS
        assert "conclusions" in _COMMON_HEADING_WORDS

    def test_references(self):
        assert "references" in _COMMON_HEADING_WORDS

    def test_introduction(self):
        assert "introduction" in _COMMON_HEADING_WORDS


class TestClassifyBlocksHeadingLevel:
    def test_numbered_heading_gets_correct_level(self):
        blocks = [
            {"type": "text", "text": "1 Introduction", "page": 0,
             "bbox": (0, 0, 100, 20), "font_size": 14, "font": "Test-Bold",
             "is_heading": True, "heading_level": 4},
        ]
        result = classify_blocks_builtin(blocks)
        assert result[0]["is_heading"] is True
        assert result[0]["heading_level"] == 1

    def test_sub_heading_level(self):
        blocks = [
            {"type": "text", "text": "3.1 Encoder and Decoder Stacks", "page": 0,
             "bbox": (0, 0, 100, 20), "font_size": 13, "font": "Test-Bold",
             "is_heading": True, "heading_level": 4},
        ]
        result = classify_blocks_builtin(blocks)
        assert result[0]["heading_level"] == 2

    def test_sub_sub_heading_level(self):
        blocks = [
            {"type": "text", "text": "3.2.1 Scaled Dot-Product Attention", "page": 0,
             "bbox": (0, 0, 100, 20), "font_size": 12, "font": "Test-Bold",
             "is_heading": True, "heading_level": 4},
        ]
        result = classify_blocks_builtin(blocks)
        assert result[0]["heading_level"] == 3


class TestShortUpperHeading:
    def test_all_caps(self):
        assert _SHORT_UPPER_HEADING.match("INTRODUÇÃO")
        assert _SHORT_UPPER_HEADING.match("OBJETIVOS")
        assert _SHORT_UPPER_HEADING.match("RESULTADOS E DISCUSSÃO")

    def test_rejects_lowercase(self):
        assert not _SHORT_UPPER_HEADING.match("introdução")
        assert not _SHORT_UPPER_HEADING.match("Objetivos")
