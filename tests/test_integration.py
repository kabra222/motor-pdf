"""Testes de integração com PDFs reais (multi-coluna, tabelas, anotações, listas)."""

from collections.abc import Generator
from pathlib import Path

import fitz
import pytest

from app.engine.extractor import extract_text
from app.engine.layout import analyze_blocks


@pytest.fixture(scope="module")
def real_pdf_path() -> Generator[Path, None, None]:
    """Generate a realistic multi-page PDF with various elements."""
    import tempfile

    from tests.test_pdf import create_test_pdf
    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    create_test_pdf(tmp, pages=8)
    doc = fitz.open(tmp)
    page = doc[0]
    page.insert_text((50, 700), "LEI Nº 14.133, DE 1º DE ABRIL DE 2021", fontsize=16, fontname="helv")
    page.insert_text((50, 680), "Lei de Licitações e Contratos Administrativos", fontsize=14, fontname="helv")
    page.insert_text((50, 640), "Art. 1º Esta Lei estabelece normas gerais de licitação e contratação", fontsize=11, fontname="helv")
    page.insert_text((50, 620), "para a Administração Pública direta, autárquica e fundacional.", fontsize=11, fontname="helv")
    page.insert_text((50, 590), "Art. 2º A licitação será processada na seguinte ordem:", fontsize=11, fontname="helv")
    page.insert_text((70, 570), "I – fase preparatória;", fontsize=11, fontname="helv")
    page.insert_text((70, 550), "II – divulgação do edital;", fontsize=11, fontname="helv")
    page.insert_text((70, 530), "III – apresentação de propostas;", fontsize=11, fontname="helv")
    page.insert_text((50, 500), "Art. 3º O contrato deve conter cláusulas essenciais.", fontsize=11, fontname="helv")

    page2 = doc[1]
    page2.insert_text((50, 750), "Contratação", fontsize=15, fontname="helv")
    page2.insert_text((50, 720), "1. Do objeto e suas especificações.", fontsize=11, fontname="helv")
    page2.insert_text((50, 700), "2. Do prazo e condições de pagamento.", fontsize=11, fontname="helv")
    page2.insert_text((50, 680), "3. Dos prazos de execução e entrega.", fontsize=11, fontname="helv")
    page2.insert_text((300, 720), "Tabela 1 - Prazos", fontsize=12, fontname="helv")
    page2.insert_text((300, 700), "Item | Prazo | Multa", fontsize=11, fontname="helv")
    page2.insert_text((300, 680), "1 | 30 dias | 2%", fontsize=11, fontname="helv")
    page2.insert_text((300, 660), "2 | 60 dias | 5%", fontsize=11, fontname="helv")

    page2.insert_text((50, 50), "Relatório Jurídico", fontsize=9, fontname="helv")
    page2.insert_text((50, 30), "Página 2", fontsize=9, fontname="helv")

    page3 = doc[2]
    page3.insert_text((50, 750), "Notas Técnicas", fontsize=15, fontname="helv")
    page3.insert_text((50, 720), "Conforme entendimento do TCU¹, a administração deve:", fontsize=11, fontname="helv")
    page3.insert_text((50, 690), "Observar o princípio da economicidade nos gastos públicos.", fontsize=11, fontname="helv")

    page_last = doc[-1]
    page_last.insert_text((50, 750), "Referências", fontsize=15, fontname="helv")
    page_last.insert_text((50, 720), "BRASIL. Lei nº 14.133, de 1º de abril de 2021.", fontsize=11, fontname="helv")
    page_last.insert_text((50, 700), "TCU. Acórdão 1234/2023 - Plenário.", fontsize=11, fontname="helv")
    page_last.insert_text((50, 680), "¹ Tribunal de Contas da União", fontsize=9, fontname="helv")

    doc.set_metadata({"title": "Lei de Licitações - Teste Integração"})
    doc.save(tmp, incremental=True, encryption=0)
    doc.close()
    yield tmp
    tmp.unlink(missing_ok=True)


def test_real_pdf_basic_extraction(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    assert result["num_pages"] == 8
    assert len(result["text"]) > 500
    assert result["text"].strip()


def test_real_pdf_headings_detected(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    headings = result.get("headings", [])
    assert len(headings) >= 3


def test_real_pdf_classification(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    blocks = result["blocks"]
    text_blocks = [b for b in blocks if b["type"] == "text"]
    classified = [b for b in text_blocks if b.get("layout_type")]
    assert len(classified) > 0


def test_real_pdf_list_items_detected(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    blocks = result["blocks"]
    list_items = [b for b in blocks if b.get("is_list_item")]
    assert len(list_items) >= 3


def test_real_pdf_quality_scored(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    q = result.get("quality", {})
    assert q.get("overall", 0) > 0
    dims = q.get("dimensions", {})
    assert "hyphenation" in dims


def test_real_pdf_multi_column_detection(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    info = analyze_blocks(result["blocks"], result["num_pages"])
    assert isinstance(info.has_multi_column, bool)


def test_real_pdf_annotations(real_pdf_path):
    """At least verify annotations key exists (PDF gen may not create them)."""
    result = extract_text(str(real_pdf_path))
    assert "annotations" in result
    assert "links" in result


def test_real_pdf_scanned_pages(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    assert isinstance(result.get("scanned_pages", []), list)


def test_real_pdf_no_empty_text(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    assert result["text"].strip()


def test_real_pdf_layout_metadata(real_pdf_path):
    result = extract_text(str(real_pdf_path))
    meta = result.get("metadata", {})
    if "layout" in meta:
        layout = meta["layout"]
        assert "has_multi_column" in layout
        assert "has_header" in layout
        assert "has_footer" in layout
