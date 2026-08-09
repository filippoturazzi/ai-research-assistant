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


def test_no_match_excludes_irrelevant_docs():
    """Verify that non-matching documents (zero query-term overlap) are excluded from results."""
    idx = BM25Index([
        "the transformer architecture uses attention",
        "convolutional networks process images",
        "reinforcement learning maximizes reward",
    ])
    # Query "transformer attention" should only match doc 0
    hits = idx.search("transformer attention", k=10)
    # Verify only doc 0 is returned (docs 1 and 2 have no vocabulary overlap)
    assert len(hits) == 1
    assert hits[0][0] == 0
    assert hits[0][1] > 0
