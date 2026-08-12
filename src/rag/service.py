import hashlib
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.config import GENERATION_MODEL
from rag.errors import (DocumentNotFoundError, DuplicateDocumentError,
                        EmptyIndexError, ExtractionError)
from rag.feedback.db import FeedbackDB
from rag.generation.generator import generate_answer
from rag.generation.groq_chat import GroqChat
from rag.generation.rewriter import rewrite_query
from rag.generation.suggestions import suggest_questions
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
        # (base fingerprint, {language: questions}) — a new fingerprint drops the
        # whole language map, so the cache only ever holds the current base.
        self._suggestions_cache: tuple[str, dict[str, list[str]]] = ("", {})

    def ask(self, question: str, history: list[dict] | None = None,
            language: str = "en") -> AskResult:
        if not self.store.chunks:
            raise EmptyIndexError(
                "The knowledge base is empty — add documents on the Documents page first."
            )
        start = time.perf_counter()
        rewritten = rewrite_query(self.chat, question, history or [], language)
        retrieved = self.retriever.retrieve(rewritten)
        answer = generate_answer(self.chat, question,
                                 [r.chunk for r in retrieved], language)
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
            raise ValueError("Invalid file name.")
        if Path(safe_name).stem in self.store.doc_ids():
            raise DuplicateDocumentError(f"Document '{Path(safe_name).stem}' is already indexed.")
        path = self.documents_dir / safe_name
        path.write_bytes(pdf_bytes)
        added = ingest_pdf(path, self.store, self.embedder)
        self.store.save(self.index_dir)
        return added

    def add_documents(self, files: list[tuple[str, bytes]]) -> list[dict]:
        results = []
        for filename, data in files:
            doc_id = Path(filename).stem
            try:
                added = self.add_document(data, filename)
                results.append({"filename": filename, "doc_id": doc_id,
                                "chunks_added": added, "error": None})
            except (DuplicateDocumentError, ExtractionError, ValueError) as exc:
                results.append({"filename": filename, "doc_id": doc_id,
                                "chunks_added": 0, "error": str(exc)})
        return results

    def remove_document(self, doc_id: str) -> int:
        if doc_id not in self.store.doc_ids():
            raise DocumentNotFoundError(f"Document '{doc_id}' is not indexed.")
        removed = self.store.remove(doc_id)
        pdf = self.documents_dir / f"{Path(doc_id).name}.pdf"
        if pdf.exists():
            pdf.unlink()
        self.store.save(self.index_dir)
        return removed

    def reset_documents(self) -> int:
        removed = len(self.store.chunks)
        self.store.clear()
        if self.documents_dir.exists():
            for pdf in self.documents_dir.glob("*.pdf"):
                pdf.unlink()
        self.store.save(self.index_dir)
        return removed

    def restore_default_documents(self, fetch=None) -> dict:
        if fetch is None:
            from rag.ingestion.default_papers import fetch_default_papers
            fetch = fetch_default_papers
        self.reset_documents()
        documents_added = 0
        chunks_added = 0
        for filename, data in fetch():
            chunks_added += self.add_document(data, filename)
            documents_added += 1
        return {"documents_added": documents_added, "chunks_added": chunks_added}

    def feedback(self, interaction_id: int, rating: int, comment: str | None = None) -> None:
        self.db.add_feedback(interaction_id, rating, comment)

    def metrics(self) -> dict:
        return self.db.metrics()

    def documents(self) -> list[dict]:
        counts = Counter((c.doc_id, c.doc_title) for c in self.store.chunks)
        return [{"doc_id": doc_id, "doc_title": title, "chunks": n}
                for (doc_id, title), n in sorted(counts.items())]

    def _base_fingerprint(self) -> str:
        counts = Counter(c.doc_id for c in self.store.chunks)
        return hashlib.md5(repr(sorted(counts.items())).encode()).hexdigest()

    def suggested_questions(self, language: str = "en") -> list[str]:
        fingerprint = self._base_fingerprint()
        cached_fingerprint, by_language = self._suggestions_cache
        if cached_fingerprint != fingerprint:
            by_language = {}
            self._suggestions_cache = (fingerprint, by_language)
        if language not in by_language:
            by_language[language] = suggest_questions(self.chat, self.store.chunks,
                                                      language)
        return by_language[language]
