import pytest

from rag.errors import ExtractionError
from rag.ingestion.pdf_extractor import extract_pages


def test_extracts_pages_with_numbers(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert [p for p, _ in pages] == [1, 2]
    assert "attention" in pages[0][1].lower()
    assert "bidirectional" in pages[1][1].lower()


def test_invalid_pdf_raises(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(ExtractionError):
        extract_pages(bad)
