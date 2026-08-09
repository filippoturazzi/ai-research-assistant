import numpy as np

from rag.config import RERANKER_MODEL, TOP_K
from rag.models import Chunk


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder  # import tardio: pesado
            model = CrossEncoder(model_name)
        self._model = model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        scores = np.asarray(self._model.predict([(query, c.text) for c in chunks]), dtype="float32")
        order = np.argsort(scores)[::-1][:top_k]
        return [(chunks[i], float(scores[i])) for i in order]
