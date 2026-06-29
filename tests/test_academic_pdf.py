from __future__ import annotations

from app.engine.extractor import extract_text


class TestAcademicPDF:
    def test_extracts_text(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        text = result.get("text", "")
        assert len(text) > 10000

    def test_detects_headings(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        headings = result.get("headings", [])
        assert len(headings) >= 5

    def test_detects_tables(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        tables = result.get("tables", [])
        assert len(tables) >= 3

    def test_quality_scored(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        quality = result.get("quality", {})
        assert quality.get("overall", 0) > 0
        assert "dimensions" in quality

    def test_math_notation_detected(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        text = result.get("text", "")
        from app.engine.math import has_math_patterns
        assert has_math_patterns(text)

    def test_boilerplate_removed(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        text = result.get("text", "")
        for phrase in ["Continue from your last visit?", "FULLSCREEN", "COPY LINK"]:
            assert phrase not in text

    def test_blocks_not_empty(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        blocks = result.get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        assert len(text_blocks) > 0

    def test_tables_have_content(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        tables = result.get("tables", [])
        non_empty = [t for t in tables if len(t) > 50 and "|" in t]
        assert len(non_empty) >= 3

    def test_classification_diversity(self, academic_pdf_path):
        result = extract_text(str(academic_pdf_path))
        blocks = result.get("blocks", [])
        types = set(b.get("layout_type", "") for b in blocks if b.get("type") == "text")
        assert "heading" in types
        assert "paragraph" in types
