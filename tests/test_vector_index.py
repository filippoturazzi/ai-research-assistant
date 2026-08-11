import numpy as np

from rag.retrieval.vector_index import VectorIndex


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def test_search_returns_nearest_first():
    idx = VectorIndex(dim=4)
    idx.add(np.stack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]), _unit([1, 1, 0, 0])]))
    hits = idx.search(_unit([1, 0.1, 0, 0]), k=2)
    assert hits[0][0] == 0
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]


def test_search_empty_index():
    assert VectorIndex(dim=4).search(_unit([1, 0, 0, 0]), k=5) == []


def test_reconstruct_all_returns_stored_vectors():
    idx = VectorIndex(dim=4)
    vectors = np.stack([_unit([1, 0, 0, 0]), _unit([0, 1, 1, 0])])
    idx.add(vectors)
    out = idx.reconstruct_all()
    assert out.shape == (2, 4)
    assert np.allclose(out, vectors)


def test_reconstruct_all_empty_index():
    assert VectorIndex(dim=4).reconstruct_all().shape == (0, 4)


def test_save_and_load(tmp_path):
    idx = VectorIndex(dim=4)
    idx.add(np.stack([_unit([0, 0, 1, 0])]))
    idx.save(tmp_path / "v.faiss")
    loaded = VectorIndex.load(tmp_path / "v.faiss")
    assert loaded.size == 1
    assert loaded.search(_unit([0, 0, 1, 0]), k=1)[0][0] == 0
