from __future__ import annotations

from app.engine.boilerplate import (
    filter_boilerplate,
    filter_boilerplate_from_blocks,
    is_boilerplate_line,
    remove_page_markers_inline,
)


class TestBoilerplateDetection:
    def test_continue_from_last_visit(self):
        assert is_boilerplate_line("Continue from your last visit?")
        assert is_boilerplate_line("  Continue from your last visit?  ")

    def test_fullscreen(self):
        assert is_boilerplate_line("FULLSCREEN")

    def test_copy_link(self):
        assert is_boilerplate_line("COPY LINK")

    def test_continue_reading(self):
        assert is_boilerplate_line("CONTINUE READING")

    def test_link(self):
        assert is_boilerplate_line("LINK")

    def test_page_fraction(self):
        assert is_boilerplate_line("1/8")
        assert is_boilerplate_line("12/15")
        assert is_boilerplate_line(" 3/10 ")

    def test_page_of(self):
        assert is_boilerplate_line("Page 3 of 10")
        assert is_boilerplate_line("Page 1 of 5")

    def test_valid_content_not_filtered(self):
        assert not is_boilerplate_line("O presente relatorio descreve...")
        assert not is_boilerplate_line("9.8 How poor countries get stuck")
        assert not is_boilerplate_line("Introdução")
        assert not is_boilerplate_line("1. Introdução")
        assert not is_boilerplate_line("")

    def test_heading_not_filtered(self):
        assert not is_boilerplate_line("## Título")
        assert not is_boilerplate_line("### Seção 1.1")


class TestFilterBoilerplate:
    def test_removes_boilerplate_lines(self):
        text = "Título\n\nContinue from your last visit?\n\nConteúdo.\n\nCOPY LINK\n\n1/8\n\nFim."
        filtered = filter_boilerplate(text)
        assert "Continue from your last visit?" not in filtered
        assert "COPY LINK" not in filtered
        assert "1/8" not in filtered
        assert "Título" in filtered
        assert "Conteúdo." in filtered
        assert "Fim." in filtered

    def test_preserves_empty_text(self):
        assert filter_boilerplate("") == ""

    def test_preserves_text_without_boilerplate(self):
        text = "Apenas conteúdo legítimo.\n\nMais conteúdo."
        assert filter_boilerplate(text) == text


class TestFilterBoilerplateFromBlocks:
    def test_removes_boilerplate_blocks(self):
        blocks = [
            {"text": "Título", "type": "text"},
            {"text": "Continue from your last visit?", "type": "text"},
            {"text": "Conteúdo", "type": "text"},
            {"text": "1/8", "type": "text"},
        ]
        result = filter_boilerplate_from_blocks(blocks)
        assert len(result) == 2
        assert result[0]["text"] == "Título"
        assert result[1]["text"] == "Conteúdo"

    def test_preserves_non_text_blocks(self):
        blocks = [
            {"type": "image", "width": 100},
            {"text": "Continue from your last visit?", "type": "text"},
        ]
        result = filter_boilerplate_from_blocks(blocks)
        assert len(result) == 1
        assert result[0]["type"] == "image"


class TestRemovePageMarkersInline:
    def test_removes_inline_markers(self):
        assert "abc" in remove_page_markers_inline("abc 1/8 def")

    def test_preserves_text_without_markers(self):
        assert remove_page_markers_inline("Texto normal") == "Texto normal"
