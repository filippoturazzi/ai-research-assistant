"""Builds the index (FAISS + chunks.json) from data/documents/*.pdf."""
import sys

from rag.config import DOCUMENTS_DIR, INDEX_DIR
from rag.errors import ExtractionError
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.embedder import Embedder
from rag.retrieval.store import IndexStore


def main() -> None:
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in '{DOCUMENTS_DIR}'. Run first: python scripts/download_papers.py")

    print("Loading embedding model...")
    embedder = Embedder()
    store = IndexStore()

    for pdf in pdfs:
        try:
            added = ingest_pdf(pdf, store, embedder)
            print(f"[ok] {pdf.name}: {added} chunks")
        except ExtractionError as exc:
            print(f"[error] {pdf.name}: {exc}")

    store.save(INDEX_DIR)
    print(f"Index saved to '{INDEX_DIR}' ({len(store.chunks)} chunks).")


if __name__ == "__main__":
    main()
