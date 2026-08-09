from rag.models import Chunk
from rag.retrieval.reranker import Reranker


def _chunk(i, text):
    return Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title="D", page=1, position=i, text=text)


class FakeCrossEncoder:
    def predict(self, pairs):
        # pontua mais alto quando o chunk contém a query
        return [1.0 if p[0] in p[1] else 0.0 for p in pairs]


def test_rerank_orders_by_score():
    chunks = [_chunk(0, "nothing here"), _chunk(1, "the query appears: attention")]
    out = Reranker(model=FakeCrossEncoder()).rerank("attention", chunks, top_k=2)
    assert out[0][0].position == 1
    assert out[0][1] > out[1][1]


def test_rerank_truncates_to_top_k():
    chunks = [_chunk(i, "attention text") for i in range(10)]
    out = Reranker(model=FakeCrossEncoder()).rerank("attention", chunks, top_k=5)
    assert len(out) == 5


def test_rerank_empty():
    assert Reranker(model=FakeCrossEncoder()).rerank("q", [], top_k=5) == []
