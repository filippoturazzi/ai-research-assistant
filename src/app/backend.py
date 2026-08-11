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
    import sys
    import threading
    import time
    from dataclasses import asdict
    from pathlib import Path

    from app.api_client import ApiConnectionError, ApiError

    # NOTE: no top-level `rag` imports here. Streamlit Cloud reloads app/
    # modules on deploy but never the rag package (outside the watched app
    # folder), so any rag classes bound at import time could belong to a
    # previous deploy. rag is imported inside functions, always after
    # _ensure_fresh_rag() has synced sys.modules with the code on disk.

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
        if root is None:
            import rag
            root = Path(rag.__file__).parent
        digest = hashlib.md5()
        for path in sorted(root.rglob("*.py")):
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _ensure_fresh_rag() -> str:
        # The marker lives on `sys` because it must survive reloads of this
        # module; when the rag sources on disk differ from what this process
        # loaded, drop the package so the next import picks up the new code.
        version = _code_version()
        if getattr(sys, "_rag_loaded_version", None) != version:
            for name in [m for m in list(sys.modules)
                         if m == "rag" or m.startswith("rag.")]:
                del sys.modules[name]
            # also drop cached resources: any cached service was built from
            # the module generation just purged
            st.cache_resource.clear()
            sys._rag_loaded_version = version
        return version

    # code_version (not "version"): the rename shifts this function's cache
    # hash, orphaning entries a pre-purge deploy built from stale modules
    @st.cache_resource(show_spinner="Loading models and index (first visit only)...")
    def _cached_service(code_version: str):
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
        return _cached_service(_ensure_fresh_rag())

    def _service_or_api_error():
        _ensure_fresh_rag()
        from rag.errors import GenerationError, IndexNotFoundError
        try:
            return _build_service()
        except (GenerationError, IndexNotFoundError) as exc:
            raise ApiError(str(exc)) from exc

    def ask(question: str, history: list[dict], language: str = "en") -> dict:
        _check_rate()
        if len(question) > 500:
            raise ApiError("Question too long (max 500 characters).")
        service = _service_or_api_error()
        from rag.errors import GenerationError
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
        from rag.errors import DocumentNotFoundError
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
        from rag.errors import (DownloadError, DuplicateDocumentError,
                                ExtractionError)
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
