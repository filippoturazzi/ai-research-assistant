from rag.retrieval.bm25_index import BM25Index


def test_lexical_match_ranks_first():
    idx = BM25Index([
        "the transformer architecture uses attention",
        "convolutional networks process images",
        "reinforcement learning maximizes reward",
    ])
    hits = idx.search("transformer attention", k=2)
    assert hits[0][0] == 0


def test_empty_index_returns_nothing():
    assert BM25Index([]).search("anything", k=5) == []


def test_add_rebuilds():
    idx = BM25Index(["first document about cats"])
    idx.add(["second document about faiss indexes"])
    hits = idx.search("faiss", k=1)
    assert hits[0][0] == 1
