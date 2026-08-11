import numpy as np
import pytest

from rag.errors import IndexNotFoundError
from rag.models import Chunk
from rag.retrieval.store import IndexStore


def _chunk(doc, i, text):
    return Chunk(chunk_id=f"{doc}:{i}", doc_id=doc, doc_title=doc.title(), page=1, position=i, text=text)


def _vecs(n):
    out = np.eye(4, dtype="float32")[:n]
    return out


def test_add_keeps_positions_aligned():
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "cats"), _chunk("b", 0, "faiss index")], _vecs(2))
    assert store.vectors.size == 2
    assert store.bm25.search("faiss", k=1)[0][0] == 1
    assert store.doc_ids() == {"a", "b"}


def test_save_load_roundtrip(tmp_path):
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "hello world")], _vecs(1))
    store.save(tmp_path)
    loaded = IndexStore.load(tmp_path)
    assert len(loaded.chunks) == 1
    assert loaded.chunks[0].text == "hello world"
    assert loaded.vectors.size == 1
    assert loaded.bm25.search("hello", k=1)[0][0] == 0


def test_load_missing_raises(tmp_path):
    with pytest.raises(IndexNotFoundError):
        IndexStore.load(tmp_path / "nope")


def test_remove_middle_doc_keeps_alignment():
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "cats purr"), _chunk("b", 0, "faiss index"),
               _chunk("c", 0, "dogs bark")], _vecs(3))

    removed = store.remove("b")

    assert removed == 1
    assert store.doc_ids() == {"a", "c"}
    assert store.vectors.size == 2
    # vector row 1 must now be doc "c"'s original vector (basis e3)
    hits = store.vectors.search(np.eye(4, dtype="float32")[2], k=1)
    assert hits[0][0] == 1
    assert store.chunks[1].doc_id == "c"
    # BM25 rebuilt: "dogs" must hit the new position 1
    assert store.bm25.search("dogs", k=1)[0][0] == 1


def test_remove_absent_doc_returns_zero():
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "hello")], _vecs(1))
    assert store.remove("nope") == 0
    assert len(store.chunks) == 1


def test_remove_last_doc_leaves_empty_store():
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "hello"), _chunk("a", 1, "world")], _vecs(2))
    assert store.remove("a") == 2
    assert store.chunks == []
    assert store.vectors.size == 0


def test_clear_empties_everything():
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "hello"), _chunk("b", 0, "world")], _vecs(2))
    store.clear()
    assert store.chunks == []
    assert store.vectors.size == 0
    assert store.doc_ids() == set()


def test_save_load_roundtrip_after_remove(tmp_path):
    store = IndexStore(dim=4)
    store.add([_chunk("a", 0, "hello world"), _chunk("b", 0, "faiss index")], _vecs(2))
    store.remove("a")
    store.save(tmp_path)
    loaded = IndexStore.load(tmp_path)
    assert loaded.doc_ids() == {"b"}
    assert loaded.vectors.size == 1
    assert loaded.bm25.search("faiss", k=1)[0][0] == 0


def test_add_mismatched_lengths_raises():
    store = IndexStore(dim=4)
    with pytest.raises(ValueError, match="chunks.*and vectors.*must have the same length"):
        store.add([_chunk("a", 0, "text1"), _chunk("b", 0, "text2")], _vecs(1))
    assert len(store.chunks) == 0
    assert store.vectors.size == 0
