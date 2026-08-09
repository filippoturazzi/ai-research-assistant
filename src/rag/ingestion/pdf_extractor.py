from pathlib import Path

from pypdf import PdfReader

from rag.errors import ExtractionError


def extract_pages(path: Path) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(str(path))
        pages: list[tuple[int, str]] = []
        for number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((number, text))
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Não consegui ler o PDF '{path.name}': {exc}") from exc
    if not pages:
        raise ExtractionError(f"Nenhum texto extraível em '{path.name}'.")
    return pages
