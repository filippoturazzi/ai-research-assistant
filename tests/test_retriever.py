import numpy as np

from rag.models import Chunk
from rag.retrieval.retriever import HybridRetriever, RetrievedChunk
from rag.retrieval.store import IndexStore


class PassthroughReranker:
    def rerank(self, query, chunks, top_k):
        return [(c, float(len(chunks) - i)) for i, c in enumerate(chunks[:top_k])]


class OneHotEmbedder:
    """Query 'dim0'..'dim3' vira o eixo correspondente."""

    def embed_query(self, text):
        v = np.zeros(4, dtype="float32")
        v[int(text[-1])] = 1.0
        return v


def _store():
    texts = ["faiss vector search", "bm25 lexical search", "transformers attention", "cats and dogs"]
    chunks = [Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title="D", page=1, position=i, text=t)
              for i, t in enumerate(texts)]
    store = IndexStore(dim=4)
    store.add(chunks, np.eye(4, dtype="float32"))
    return store


def test_hybrid_combines_vector_and_lexical():
    retriever = HybridRetriever(_store(), OneHotEmbedder(), PassthroughReranker(), top_k=3)
    # vetorial aponta para posição 0 ('dim0'); lexical acha 'bm25' na posição 1
    results = retriever.retrieve("bm25 dim0")
    positions = [r.chunk.position for r in results]
    assert 0 in positions and 1 in positions
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_respects_top_k():
    retriever = HybridRetriever(_store(), OneHotEmbedder(), PassthroughReranker(), top_k=2)
    assert len(retriever.retrieve("search dim0")) == 2
