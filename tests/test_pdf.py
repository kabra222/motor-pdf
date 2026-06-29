"""Generate a minimal test PDF for CI/CD pipeline."""
from pathlib import Path


def create_test_pdf(path: str | Path, pages: int = 3) -> Path:
    """Create a minimal PDF with varied layout for testing."""
    try:
        from fpdf import FPDF
    except ImportError:
        return _create_pymupdf_pdf(path, pages)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(text="Capítulo 1 - Introdução", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Este é um parágrafo de exemplo com algumas palavras.", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(text="Outro parágrafo com mais texto para teste.", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(text="E mais um parágrafo para completar a pagina.", new_x="LMARGIN", new_y="NEXT")

    for _ in range(pages - 1):
        pdf.add_page()
        pdf.set_font("Helvetica", size=14)
        pdf.cell(text="Seção 1.1", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=12)
        pdf.cell(text="Conteúdo da seção com texto suficiente para teste automatizado.", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(text="Parágrafo final desta página.", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(path))
    return Path(path)


def _create_pymupdf_pdf(path: str | Path, pages: int = 3) -> Path:
    import fitz
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        text = f"Página {i+1}\n\n"
        if i == 0:
            text += "Capítulo 1 - Introdução\n\nEste é um parágrafo de exemplo."
        else:
            text += "Seção 1.1\n\nConteúdo da seção para teste."
        page.insert_text((50, 50), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return Path(path)
