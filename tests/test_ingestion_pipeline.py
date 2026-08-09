import numpy as np
import pytest

from rag.errors import DuplicateDocumentError
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.store import IndexStore


class FakeEmbedder:
    def embed_texts(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


def test_ingest_adds_chunks(sample_pdf):
    store = IndexStore(dim=4)
    added = ingest_pdf(sample_pdf, store, FakeEmbedder())
    assert added == len(store.chunks) > 0
    assert store.chunks[0].doc_id == "sample"
    assert store.chunks[0].doc_title == "Sample"
    assert store.vectors.size == added


def test_duplicate_raises(sample_pdf):
    store = IndexStore(dim=4)
    ingest_pdf(sample_pdf, store, FakeEmbedder())
    with pytest.raises(DuplicateDocumentError):
        ingest_pdf(sample_pdf, store, FakeEmbedder())
