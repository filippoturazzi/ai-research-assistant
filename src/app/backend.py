"""Backend switch for the UI: HTTP (default) or embedded RAG core.

Set BACKEND_MODE=embedded (env var or st.secrets) to run the RAG service
in-process — used by the hosted demo on Streamlit Community Cloud, where a
separate FastAPI process is not available.
"""
import os


def _mode() -> str:
    mode = os.environ.get("BACKEND_MODE", "http")
    try:
        import streamlit as st
        mode = st.secrets.get("BACKEND_MODE", mode)
    except Exception:
        pass
    return mode


if _mode() != "embedded":
    from app.api_client import (ApiConnectionError, ApiError, ask, documents,
                                metrics, send_feedback, upload)
else:
    import streamlit as st
    import threading
    import time
    from dataclasses import asdict
    from pathlib import Path

    from app.api_client import ApiConnectionError, ApiError
    from rag.errors import DuplicateDocumentError, ExtractionError, GenerationError

    _RATE_LIMIT = 10
    _RATE_WINDOW_S = 60
    _hits: list[float] = []
    _hits_lock = threading.Lock()

    def _check_rate() -> None:
        now = time.monotonic()
        with _hits_lock:
            _hits[:] = [t for t in _hits if now - t < _RATE_WINDOW_S]
            if len(_hits) >= _RATE_LIMIT:
                raise ApiError("Rate limit exceeded — try again in a minute.")
            _hits.append(now)

    def _bridge_secrets() -> None:
        try:
            if "GROQ_API_KEY" in st.secrets:
                os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
        except Exception:
            pass

    @st.cache_resource(show_spinner="Loading models and index (first visit only)...")
    def _cached_service():
        _bridge_secrets()
        from rag.config import DB_PATH, DOCUMENTS_DIR, INDEX_DIR
        from rag.feedback.db import FeedbackDB
        from rag.generation.groq_chat import GroqChat
        from rag.retrieval.embedder import Embedder
        from rag.retrieval.reranker import Reranker
        from rag.retrieval.store import IndexStore
        from rag.service import RAGService

        return RAGService(store=IndexStore.load(INDEX_DIR), embedder=Embedder(),
                          reranker=Reranker(), chat=GroqChat(), db=FeedbackDB(DB_PATH),
                          index_dir=INDEX_DIR, documents_dir=DOCUMENTS_DIR)

    def _build_service():
        return _cached_service()

    def ask(question: str, history: list[dict], language: str = "en") -> dict:
        _check_rate()
        if len(question) > 500:
            raise ApiError("Question too long (max 500 characters).")
        try:
            result = _build_service().ask(question, history, language)
        except GenerationError as exc:
            raise ApiError(str(exc)) from exc
        return {"interaction_id": result.interaction_id, "answer": result.answer,
                "rewritten_query": result.rewritten_query,
                "sources": [asdict(s) for s in result.sources]}

    def upload(filename: str, data: bytes) -> dict:
        _check_rate()
        try:
            added = _build_service().add_document(data, filename)
        except (DuplicateDocumentError, ExtractionError, ValueError) as exc:
            raise ApiError(str(exc)) from exc
        return {"doc_id": Path(filename).stem, "chunks_added": added}

    def send_feedback(interaction_id: int, rating: int) -> dict:
        _build_service().feedback(interaction_id, rating)
        return {"ok": True}

    def metrics() -> dict:
        return _build_service().metrics()

    def documents() -> list:
        return _build_service().documents()
