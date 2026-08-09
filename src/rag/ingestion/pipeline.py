import re
from pathlib import Path

from rag.errors import DuplicateDocumentError
from rag.ingestion.chunker import chunk_pages
from rag.ingestion.pdf_extractor import extract_pages


def _title_from_stem(stem: str) -> str:
    return re.sub(r"[_-]+", " ", stem).strip().title()


def ingest_pdf(path: Path, store, embedder) -> int:
    doc_id = path.stem
    if doc_id in store.doc_ids():
        raise DuplicateDocumentError(f"Documento '{doc_id}' já está indexado.")
    pages = extract_pages(path)
    chunks = chunk_pages(pages, doc_id=doc_id, doc_title=_title_from_stem(doc_id))
    if not chunks:
        return 0
    vectors = embedder.embed_texts([c.text for c in chunks])
    store.add(chunks, vectors)
    return len(chunks)
