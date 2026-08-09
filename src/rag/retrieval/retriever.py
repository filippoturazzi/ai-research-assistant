from dataclasses import dataclass

from rag.config import CANDIDATES_PER_INDEX, RERANK_CANDIDATES, TOP_K
from rag.models import Chunk
from rag.retrieval.fusion import reciprocal_rank_fusion


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, store, embedder, reranker,
                 candidates_per_index: int = CANDIDATES_PER_INDEX,
                 rerank_candidates: int = RERANK_CANDIDATES,
                 top_k: int = TOP_K):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.candidates_per_index = candidates_per_index
        self.rerank_candidates = rerank_candidates
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        vector_hits = self.store.vectors.search(
            self.embedder.embed_query(query), self.candidates_per_index)
        lexical_hits = self.store.bm25.search(query, self.candidates_per_index)
        fused = reciprocal_rank_fusion(
            [[i for i, _ in vector_hits], [i for i, _ in lexical_hits]])
        candidates = [self.store.chunks[i] for i, _ in fused[:self.rerank_candidates]]
        ranked = self.reranker.rerank(query, candidates, top_k=self.top_k)
        return [RetrievedChunk(chunk=c, score=s) for c, s in ranked]
