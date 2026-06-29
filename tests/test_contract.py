from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.engine.extractor import extract_text
from app.engine.formatter import format_result
from app.models import (
    Chunk,
    DocumentJob,
    ErrorResponse,
    ExtractionResult,
    FormatType,
    HeadingInfo,
    JobResult,
    JobStatus,
    QualityDimension,
    TextBlock,
    ImageBlock,
    TableBlock,
)


def _make_extraction_data() -> dict:
    return {
        "filename": "test.pdf",
        "text": "Conteúdo de exemplo para teste.",
        "chunks": [
            {
                "index": 0,
                "text": "Conteúdo de exemplo para teste.",
                "page": 1,
                "section": None,
                "heading": None,
                "tokens": 5,
                "depth": 0,
                "start_char": 0,
                "end_char": 35,
            }
        ],
        "blocks": [
            {
                "type": "text",
                "text": "Conteúdo de exemplo para teste.",
                "page": 1,
                "bbox": [50, 700, 500, 720],
                "font_size": 12,
                "font": "Helvetica",
                "is_heading": False,
                "heading_level": None,
                "is_list_item": False,
                "list_type": None,
                "layout_type": "paragraph",
                "is_noise": False,
            }
        ],
        "headings": [],
        "tables": [],
        "images": [],
        "metadata": {
            "title": "Teste",
            "author": "",
            "subject": "",
            "keywords": "",
            "encrypted": False,
        },
        "num_pages": 1,
        "num_chunks": 1,
        "scanned_pages": [],
        "quality": None,
        "classified_count": 1,
        "annotations": [],
        "links": [],
    }


class TestExtractionResultContract:
    def test_full_result_validates(self):
        data = _make_extraction_data()
        result = ExtractionResult(**data)
        assert result.filename == "test.pdf"
        assert result.num_pages == 1
        assert result.num_chunks == 1
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "Conteúdo de exemplo para teste."
        assert isinstance(result.chunks[0], Chunk)

    def test_missing_optional_fields_default(self):
        data = _make_extraction_data()
        del data["annotations"]
        del data["links"]
        result = ExtractionResult(**data)
        assert result.annotations == []
        assert result.links == []

    def test_null_quality(self):
        data = _make_extraction_data()
        data["quality"] = None
        result = ExtractionResult(**data)
        assert result.quality is None

    def test_invalid_type_raises(self):
        data = _make_extraction_data()
        data["num_pages"] = "invalid"
        with pytest.raises(ValidationError):
            ExtractionResult(**data)

    def test_discriminated_block_types(self):
        data = _make_extraction_data()
        result = ExtractionResult(**data)
        for b in result.blocks:
            assert b.type in ("text", "image", "table")
            if b.type == "text":
                assert hasattr(b, "font_size")


class TestChunkContract:
    def test_minimal_chunk(self):
        c = Chunk(index=0, text="texto", page=1)
        assert c.tokens == 0
        assert c.depth == 0

    def test_chunk_fields(self):
        c = Chunk(
            index=0,
            text="texto longo" * 10,
            page=2,
            section="Capítulo 1",
            heading="Introdução",
            tokens=25,
            depth=1,
            start_char=0,
            end_char=100,
        )
        assert c.section == "Capítulo 1"
        assert c.heading == "Introdução"
        assert c.tokens == 25


class TestTextBlockContract:
    def test_defaults(self):
        b = TextBlock(text="exemplo", page=1)
        assert b.type == "text"
        assert b.font_size == 12
        assert b.is_heading is False
        assert b.is_noise is False

    def test_heading_block(self):
        b = TextBlock(
            text="Capítulo 1",
            page=1,
            is_heading=True,
            heading_level=1,
            layout_type="heading",
        )
        assert b.is_heading
        assert b.heading_level == 1


class TestHeadingInfoContract:
    def test_heading_info(self):
        h = HeadingInfo(level=2, text="Seção 1.1", page=3, bbox=(0, 0, 100, 20))
        assert h.level == 2
        assert h.text == "Seção 1.1"


class TestDocumentJobContract:
    def test_initial_status(self):
        job = DocumentJob(id="abc123")
        assert job.status == JobStatus.processing
        assert job.progress == 0

    def test_with_result(self):
        ext = ExtractionResult(**_make_extraction_data())
        job = DocumentJob(id="abc123", status="done", result=ext)
        assert job.status == JobStatus.done
        assert job.result is not None


class TestJobResultContract:
    def test_job_result(self):
        r = JobResult(id="abc", status="done")
        assert r.status == JobStatus.done


class TestErrorResponseContract:
    def test_minimal(self):
        e = ErrorResponse(error="Algo deu errado")
        assert e.error == "Algo deu errado"
        assert e.detail is None

    def test_with_detail(self):
        e = ErrorResponse(error="Erro", detail="Detalhe do erro")
        assert e.detail == "Detalhe do erro"


class TestFormatResultContract:
    def test_markdown_output(self):
        text = "Título\n\nParágrafo."
        blocks = [
            {
                "type": "text",
                "text": "Título",
                "page": 1,
                "is_heading": True,
                "heading_level": 1,
                "layout_type": "heading",
                "bbox": [0, 0, 0, 0],
                "font_size": 18,
                "font": "Helvetica",
                "is_list_item": False,
                "list_type": None,
                "is_noise": False,
            },
            {
                "type": "text",
                "text": "Parágrafo.",
                "page": 1,
                "is_heading": False,
                "heading_level": None,
                "layout_type": "paragraph",
                "bbox": [0, 0, 0, 0],
                "font_size": 12,
                "font": "Helvetica",
                "is_list_item": False,
                "list_type": None,
                "is_noise": False,
            },
        ]
        result = format_result(text, blocks, [], [], FormatType.markdown)
        assert "#" in result

    def test_text_output_preserves(self):
        text = "Texto simples."
        result = format_result(text, [], [], [], FormatType.text)
        assert result == "Texto simples."


class TestQualityDimensionContract:
    def test_quality_dimension(self):
        d = QualityDimension(score=0.85, detail="85% de cobertura")
        assert d.score == 0.85


@pytest.fixture(scope="session")
def sample_pdf_path():
    import tempfile
    import fitz

    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((50, 750), "Título de Teste", fontsize=18)
    p.insert_text((50, 700), "Parágrafo exemplo com conteúdo relevante.", fontsize=11)
    doc.set_metadata({"title": "Documento Teste", "author": "Autor Teste"})
    doc.save(str(tmp))
    doc.close()
    return tmp


class TestExtractionPipelineContract:
    def _to_block(self, b: dict) -> TextBlock | ImageBlock | TableBlock:
        if b.get("type") == "image":
            return ImageBlock(**b)
        if b.get("type") == "table":
            return TableBlock(**b)
        return TextBlock(**b)

    def test_extract_text_schema(self, sample_pdf_path):
        result = extract_text(str(sample_pdf_path))
        validated = ExtractionResult(
            filename=sample_pdf_path.name,
            text=result["text"],
            chunks=[Chunk(**c) if isinstance(c, dict) else c for c in result.get("chunks", [])],
            blocks=[self._to_block(b) if isinstance(b, dict) else b for b in result.get("blocks", [])],
            headings=[HeadingInfo(**h) if isinstance(h, dict) else h for h in result.get("headings", [])],
            tables=result.get("tables", []),
            images=result.get("images", []),
            metadata=result.get("metadata", {}),
            num_pages=result["num_pages"],
            num_chunks=len(result.get("chunks", [])),
            scanned_pages=result.get("scanned_pages", []),
            quality=result.get("quality"),
            classified_count=result.get("classified_count", 0),
            annotations=result.get("annotations", []),
            links=result.get("links", []),
        )
        assert validated.num_pages > 0
        assert len(validated.text) > 0

    def test_extract_all_fields_present(self, sample_pdf_path):
        result = extract_text(str(sample_pdf_path))
        for field in ("text", "blocks", "headings", "tables", "metadata", "num_pages"):
            assert field in result, f"Campo obrigatório ausente: {field}"

    def test_quality_contains_dimensions(self, sample_pdf_path):
        result = extract_text(str(sample_pdf_path))
        q = result.get("quality")
        assert q is not None
        assert "overall" in q
        assert "dimensions" in q
        assert len(q["dimensions"]) >= 3
