import numpy as np

from rag.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer  # lazy import: heavy dependency
            model = SentenceTransformer(model_name)
        self._model = model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype="float32",
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
