"""Constrói o índice (FAISS + chunks.json) a partir de data/documents/*.pdf."""
import sys
from pathlib import Path

from rag.config import DOCUMENTS_DIR, INDEX_DIR
from rag.errors import ExtractionError
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.embedder import Embedder
from rag.retrieval.store import IndexStore


def main() -> None:
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nenhum PDF em '{DOCUMENTS_DIR}'. Rode antes: python scripts/download_papers.py")

    print("Carregando modelo de embeddings...")
    embedder = Embedder()
    store = IndexStore()

    for pdf in pdfs:
        try:
            added = ingest_pdf(pdf, store, embedder)
            print(f"[ok] {pdf.name}: {added} chunks")
        except ExtractionError as exc:
            print(f"[erro] {pdf.name}: {exc}")

    store.save(INDEX_DIR)
    print(f"Índice salvo em '{INDEX_DIR}' ({len(store.chunks)} chunks).")


if __name__ == "__main__":
    main()
