from app.engine.extractor import extract_text
from app.engine.layout import analyze_blocks
from app.engine.classifier import classify_blocks_builtin, filter_noise_blocks


PDF_PATH = "/Users/raje/projects/motor pdf - código/0001 - Direito_Constitucional_Pedro_Lenza_2025__capitulos_separados/Capitulo_01__NEO CONSTITUCIONALISMO.pdf"


def test_extract_basic():
    result = extract_text(PDF_PATH)
    assert result["num_pages"] > 0
    assert result["text"]
    assert len(result["blocks"]) > 0


def test_extract_quality():
    result = extract_text(PDF_PATH)
    q = result.get("quality", {})
    assert q.get("overall", 0) > 0
    dims = q.get("dimensions", {})
    assert "hyphenation" in dims
    assert "orphan_lines" in dims
    assert "line_quality" in dims


def test_extract_metadata():
    result = extract_text(PDF_PATH)
    meta = result["metadata"]
    assert isinstance(meta, dict)
    assert "layout" in meta
    layout = meta["layout"]
    assert "has_multi_column" in layout
    assert "columns" in layout


def test_headings_detected():
    result = extract_text(PDF_PATH)
    headings = result.get("headings", [])
    assert len(headings) > 0
    assert any(h.get("level", 0) >= 1 for h in headings)


def test_classification():
    result = extract_text(PDF_PATH)
    blocks = result["blocks"]
    text_blocks = [b for b in blocks if b["type"] == "text"]
    classified = [b for b in text_blocks if b.get("layout_type") is not None]
    assert len(classified) > 0


def test_noise_filtering():
    result = extract_text(PDF_PATH)
    blocks = result["blocks"]
    noise = [b for b in blocks if b.get("is_noise")]
    assert len(noise) == 0


def test_tables():
    result = extract_text(PDF_PATH)
    tables = result.get("tables", [])
    assert isinstance(tables, list)


def test_classifier_builtin():
    result = extract_text(PDF_PATH, use_ocr="")
    blocks = result["blocks"]
    classified = classify_blocks_builtin(blocks)
    assert len(classified) == len(blocks)
    text_blocks = [b for b in classified if b["type"] == "text"]
    with_type = [b for b in text_blocks if b.get("layout_type") is not None]
    assert len(with_type) > len(text_blocks) * 0.5


def test_filter_noise():
    result = extract_text(PDF_PATH)
    blocks = result["blocks"]
    filtered = filter_noise_blocks(blocks)
    assert len(filtered) <= len(blocks)


def test_bcpd_segmentation():
    from app.engine.segmenter import segment_bcpd
    result = extract_text(PDF_PATH)
    seg_result = segment_bcpd(result["text"], result["blocks"])
    assert seg_result.segments or seg_result.method in ("bcpd_no_data", "bcpd_too_short", "bcpd_unavailable")


def test_layout_analysis():
    result = extract_text(PDF_PATH)
    info = analyze_blocks(result["blocks"], num_pages=result["num_pages"])
    assert info.has_multi_column is not None
    assert info.columns is not None
    assert isinstance(info.tables, list)


def test_hyphenation_repair():
    """verify hyphens are repaired: esta-\nbelecimento -> estabelecimento"""
    from app.engine.extractor import _repair_hyphenation
    text = "esta-\nbelecimento e configura-\nse"
    fixed = _repair_hyphenation(text)
    assert "estabelecimento" in fixed
    assert "configura-se" in fixed
