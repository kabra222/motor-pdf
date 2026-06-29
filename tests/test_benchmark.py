"""Benchmarks de performance para o motor de extração."""


from app.engine.chunker import chunk_text
from app.engine.extractor import extract_text
from app.engine.segmenter import segment_bcpd


def test_extract_benchmark(benchmark, test_pdf_path):
    result = benchmark(extract_text, str(test_pdf_path))
    assert result["num_pages"] > 0
    assert result["text"]


def test_chunk_benchmark(benchmark, test_pdf_path):
    result = extract_text(str(test_pdf_path))
    benchmark(
        chunk_text,
        result["text"],
        pages_text=result.get("pages_text"),
        blocks=result.get("blocks"),
        chunk_size=2000,
        chunk_overlap=200,
    )


def test_bcpd_segment_benchmark(benchmark, test_pdf_path):
    result = extract_text(str(test_pdf_path))
    if result["text"]:
        benchmark(segment_bcpd, result["text"], result.get("blocks", []))


def test_full_pipeline_benchmark(benchmark, test_pdf_path):
    """Full pipeline: extract + chunk + segment."""

    def run():
        r = extract_text(str(test_pdf_path))
        if r["text"]:
            chunk_text(
                r["text"],
                pages_text=r.get("pages_text"),
                blocks=r.get("blocks"),
                chunk_size=2000,
                chunk_overlap=200,
            )
        return r

    result = benchmark(run)
    assert result["num_pages"] > 0
