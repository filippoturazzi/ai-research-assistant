from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.asarray(vectors, dtype="float32"))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.size)
        if k == 0:
            return []
        scores, ids = self.index.search(np.asarray([query], dtype="float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def reconstruct_all(self) -> np.ndarray:
        if self.size == 0:
            return np.empty((0, self.dim), dtype="float32")
        return self.index.reconstruct_n(0, self.size)

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        index = faiss.read_index(str(path))
        obj = cls(index.d)
        obj.index = index
        return obj
