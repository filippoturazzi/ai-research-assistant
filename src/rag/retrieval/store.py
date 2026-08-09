import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag.config import EMBEDDING_DIM
from rag.errors import IndexNotFoundError
from rag.models import Chunk
from rag.retrieval.bm25_index import BM25Index
from rag.retrieval.vector_index import VectorIndex


class IndexStore:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.chunks: list[Chunk] = []
        self.vectors = VectorIndex(dim)
        self.bm25 = BM25Index([])

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks.extend(chunks)
        self.vectors.add(vectors)
        self.bm25.add([c.text for c in chunks])

    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self.chunks}

    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        self.vectors.save(dir / "index.faiss")
        (dir / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir: Path) -> "IndexStore":
        chunks_path = dir / "chunks.json"
        faiss_path = dir / "index.faiss"
        if not chunks_path.exists() or not faiss_path.exists():
            raise IndexNotFoundError(
                f"Índice não encontrado em '{dir}'. Rode: python scripts/build_index.py"
            )
        store = cls()
        store.chunks = [Chunk(**d) for d in json.loads(chunks_path.read_text(encoding="utf-8"))]
        store.vectors = VectorIndex.load(faiss_path)
        store.bm25 = BM25Index([c.text for c in store.chunks])
        return store
