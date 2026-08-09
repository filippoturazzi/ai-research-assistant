import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile

from api.schemas import AskRequest, AskResponse, FeedbackRequest, UploadResponse
from rag.errors import DuplicateDocumentError, ExtractionError, GenerationError


def _build_real_service():
    from rag.config import DB_PATH, DOCUMENTS_DIR, INDEX_DIR
    from rag.feedback.db import FeedbackDB
    from rag.generation.groq_chat import GroqChat
    from rag.retrieval.embedder import Embedder
    from rag.retrieval.reranker import Reranker
    from rag.retrieval.store import IndexStore
    from rag.service import RAGService

    load_dotenv()
    return RAGService(
        store=IndexStore.load(INDEX_DIR),
        embedder=Embedder(),
        reranker=Reranker(),
        chat=GroqChat(),
        db=FeedbackDB(DB_PATH),
        index_dir=INDEX_DIR,
        documents_dir=DOCUMENTS_DIR,
    )


def create_app(service=None, rate_limit: int = 10, rate_window_s: int = 60) -> FastAPI:
    app = FastAPI(title="AI Research Assistant", version="0.1.0")
    app.state.service = service or _build_real_service()

    hits: dict[str, list[float]] = {}
    hits_lock = threading.Lock()

    def svc():
        return app.state.service

    def _check_rate(request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with hits_lock:
            window = [t for t in hits.get(ip, []) if now - t < rate_window_s]
            if len(window) >= rate_limit:
                hits[ip] = window
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded — try again in a minute.",
                )
            window.append(now)
            hits[ip] = window

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request):
        _check_rate(request)
        try:
            result = svc().ask(body.question,
                               [m.model_dump() for m in body.history],
                               language=body.language)
        except GenerationError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return result

    @app.post("/upload", response_model=UploadResponse)
    async def upload(file: UploadFile, request: Request):
        _check_rate(request)
        data = await file.read()
        try:
            added = svc().add_document(data, file.filename)
        except DuplicateDocumentError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except (ExtractionError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return UploadResponse(doc_id=Path(file.filename).stem, chunks_added=added)

    @app.post("/feedback")
    def feedback(body: FeedbackRequest):
        svc().feedback(body.interaction_id, body.rating, body.comment)
        return {"ok": True}

    @app.get("/metrics")
    def metrics():
        return svc().metrics()

    @app.get("/documents")
    def documents():
        return svc().documents()

    @app.get("/health")
    def health():
        docs = svc().documents()
        return {"status": "ok", "documents": len(docs),
                "chunks": sum(d["chunks"] for d in docs)}

    return app
