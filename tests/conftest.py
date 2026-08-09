import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 10, "Transformers use attention mechanisms.")
    pdf.add_page()
    pdf.multi_cell(0, 10, "BERT is a bidirectional encoder.")
    path = tmp_path / "sample.pdf"
    pdf.output(str(path))
    return path
