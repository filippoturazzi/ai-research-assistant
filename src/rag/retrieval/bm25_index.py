import re

import numpy as np
from rank_bm25 import BM25L


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self._rebuild()

    def _rebuild(self) -> None:
        corpus = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25L(corpus) if corpus else None

    def add(self, texts: list[str]) -> None:
        self._texts.extend(texts)
        self._rebuild()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # Create pairs of (index, score), filter positive, sort descending
        scored_docs = [(i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]
