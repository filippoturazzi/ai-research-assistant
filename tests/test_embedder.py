import numpy as np

from rag.retrieval.embedder import Embedder


class FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        # vetores determinísticos pelo tamanho do texto
        out = np.stack([[len(t), 1.0, 0.0] for t in texts]).astype("float32")
        if normalize_embeddings:
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


def test_embed_texts_shape_and_norm():
    emb = Embedder(model=FakeModel())
    vecs = emb.embed_texts(["abc", "de"])
    assert vecs.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_embed_query_is_1d():
    emb = Embedder(model=FakeModel())
    assert emb.embed_query("hello").shape == (3,)
