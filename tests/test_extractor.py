from app.engine.classifier import classify_blocks_builtin, filter_noise_blocks
from app.engine.extractor import extract_text
from app.engine.layout import analyze_blocks
from app.engine.segmenter import segment_bcpd


def test_extract_basic(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    assert result["num_pages"] > 0
    assert result["text"]
    assert len(result["blocks"]) > 0


def test_extract_quality(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    q = result.get("quality", {})
    assert q.get("overall", 0) > 0
    dims = q.get("dimensions", {})
    assert "hyphenation" in dims
    assert "orphan_lines" in dims
    assert "line_quality" in dims


def test_extract_metadata(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    meta = result["metadata"]
    assert isinstance(meta, dict)
    if "layout" in meta:
        layout = meta["layout"]
        assert "has_multi_column" in layout
        assert "columns" in layout


def test_headings_detected(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    headings = result.get("headings", [])
    assert isinstance(headings, list)


def test_classification(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    blocks = result["blocks"]
    text_blocks = [b for b in blocks if b["type"] == "text"]
    classified = [b for b in text_blocks if b.get("layout_type") is not None]
    assert len(classified) > 0


def test_noise_filtering(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    blocks = result["blocks"]
    noise = [b for b in blocks if b.get("is_noise")]
    assert len(noise) == 0


def test_tables(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    tables = result.get("tables", [])
    assert isinstance(tables, list)


def test_classifier_builtin(test_pdf_path):
    result = extract_text(str(test_pdf_path), use_ocr="")
    blocks = result["blocks"]
    classified = classify_blocks_builtin(blocks)
    assert len(classified) == len(blocks)
    text_blocks = [b for b in classified if b["type"] == "text"]
    with_type = [b for b in text_blocks if b.get("layout_type") is not None]
    assert len(with_type) > len(text_blocks) * 0.5


def test_filter_noise(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    blocks = result["blocks"]
    filtered = filter_noise_blocks(blocks)
    assert len(filtered) <= len(blocks)


def test_bcpd_segmentation(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    seg_result = segment_bcpd(result["text"], result["blocks"])
    assert seg_result.segments or seg_result.method in (
        "bcpd_no_data", "bcpd_too_short", "bcpd_unavailable"
    )


def test_layout_analysis(test_pdf_path):
    result = extract_text(str(test_pdf_path))
    info = analyze_blocks(result["blocks"], num_pages=result["num_pages"])
    assert info.has_multi_column is not None
    assert info.columns is not None
    assert isinstance(info.tables, list)


def test_hyphenation_repair(test_pdf_path):
    """verify hyphens are repaired: esta-\nbelecimento -> estabelecimento"""
    from app.engine.extractor import _repair_hyphenation
    text = "esta-\nbelecimento e configura-\nse"
    fixed = _repair_hyphenation(text)
    assert "estabelecimento" in fixed
    assert "configura-se" in fixed
