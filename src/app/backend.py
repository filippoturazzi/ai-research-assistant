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
                                metrics, remove_document, reset_documents,
                                restore_defaults, send_feedback, upload)
else:
    import streamlit as st
    import hashlib
    import threading
    import time
    from dataclasses import asdict
    from pathlib import Path

    from app.api_client import ApiConnectionError, ApiError
    from rag.errors import (DocumentNotFoundError, DownloadError,
                            DuplicateDocumentError, ExtractionError,
                            GenerationError, IndexNotFoundError)

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

    def _code_version(root: Path | None = None) -> str:
        # Streamlit Cloud reloads modules on deploy without restarting the
        # process, so a cached service built from older code would survive;
        # keying the cache on the rag sources forces a rebuild per deploy.
        if root is None:
            import rag
            root = Path(rag.__file__).parent
        digest = hashlib.md5()
        for path in sorted(root.rglob("*.py")):
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @st.cache_resource(show_spinner="Loading models and index (first visit only)...")
    def _cached_service(version: str):
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
        return _cached_service(_code_version())

    def _service_or_api_error():
        try:
            return _build_service()
        except (GenerationError, IndexNotFoundError) as exc:
            raise ApiError(str(exc)) from exc

    def ask(question: str, history: list[dict], language: str = "en") -> dict:
        _check_rate()
        if len(question) > 500:
            raise ApiError("Question too long (max 500 characters).")
        service = _service_or_api_error()
        try:
            result = service.ask(question, history, language)
        except GenerationError as exc:
            raise ApiError(str(exc)) from exc
        return {"interaction_id": result.interaction_id, "answer": result.answer,
                "rewritten_query": result.rewritten_query,
                "sources": [asdict(s) for s in result.sources]}

    def upload(files: list[tuple[str, bytes]]) -> dict:
        _check_rate()
        return {"results": _service_or_api_error().add_documents(files)}

    def remove_document(doc_id: str) -> dict:
        _check_rate()
        service = _service_or_api_error()
        try:
            removed = service.remove_document(doc_id)
        except DocumentNotFoundError as exc:
            raise ApiError(str(exc)) from exc
        return {"doc_id": doc_id, "chunks_removed": removed}

    def reset_documents() -> dict:
        _check_rate()
        return {"chunks_removed": _service_or_api_error().reset_documents()}

    def restore_defaults() -> dict:
        _check_rate()
        service = _service_or_api_error()
        try:
            return service.restore_default_documents()
        except (DownloadError, DuplicateDocumentError, ExtractionError) as exc:
            raise ApiError(str(exc)) from exc

    def send_feedback(interaction_id: int, rating: int) -> dict:
        _service_or_api_error().feedback(interaction_id, rating)
        return {"ok": True}

    def metrics() -> dict:
        return _service_or_api_error().metrics()

    def documents() -> list:
        return _service_or_api_error().documents()
