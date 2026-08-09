import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.config import GENERATION_MODEL
from rag.errors import DuplicateDocumentError
from rag.feedback.db import FeedbackDB
from rag.generation.generator import generate_answer
from rag.generation.groq_chat import GroqChat
from rag.generation.rewriter import rewrite_query
from rag.ingestion.pipeline import ingest_pdf
from rag.retrieval.embedder import Embedder
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import HybridRetriever
from rag.retrieval.store import IndexStore


@dataclass
class Source:
    doc_title: str
    page: int
    text: str
    score: float


@dataclass
class AskResult:
    interaction_id: int
    answer: str
    rewritten_query: str
    sources: list[Source]


class RAGService:
    def __init__(self, store: IndexStore, embedder: Embedder, reranker: Reranker,
                 chat: GroqChat, db: FeedbackDB, index_dir: Path, documents_dir: Path):
        self.store = store
        self.embedder = embedder
        self.chat = chat
        self.db = db
        self.index_dir = index_dir
        self.documents_dir = documents_dir
        self.retriever = HybridRetriever(store, embedder, reranker)

    def ask(self, question: str, history: list[dict] | None = None) -> AskResult:
        start = time.perf_counter()
        rewritten = rewrite_query(self.chat, question, history or [])
        retrieved = self.retriever.retrieve(rewritten)
        answer = generate_answer(self.chat, question, [r.chunk for r in retrieved])
        sources = [Source(doc_title=r.chunk.doc_title, page=r.chunk.page,
                          text=r.chunk.text, score=r.score) for r in retrieved]
        latency_ms = int((time.perf_counter() - start) * 1000)
        interaction_id = self.db.log_interaction(
            query=question, rewritten_query=rewritten, answer=answer,
            sources=[asdict(s) for s in sources], model=GENERATION_MODEL,
            latency_ms=latency_ms,
        )
        return AskResult(interaction_id=interaction_id, answer=answer,
                         rewritten_query=rewritten, sources=sources)

    def add_document(self, pdf_bytes: bytes, filename: str) -> int:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("Nome de arquivo inválido.")
        if Path(safe_name).stem in self.store.doc_ids():
            raise DuplicateDocumentError(f"Documento '{Path(safe_name).stem}' já está indexado.")
        path = self.documents_dir / safe_name
        path.write_bytes(pdf_bytes)
        added = ingest_pdf(path, self.store, self.embedder)
        self.store.save(self.index_dir)
        return added

    def feedback(self, interaction_id: int, rating: int, comment: str | None = None) -> None:
        self.db.add_feedback(interaction_id, rating, comment)

    def metrics(self) -> dict:
        return self.db.metrics()

    def documents(self) -> list[dict]:
        counts = Counter((c.doc_id, c.doc_title) for c in self.store.chunks)
        return [{"doc_id": doc_id, "doc_title": title, "chunks": n}
                for (doc_id, title), n in sorted(counts.items())]
