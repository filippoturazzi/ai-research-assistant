import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from rag.errors import (DocumentNotFoundError, DownloadError,
                        DuplicateDocumentError, ExtractionError, GenerationError)
from rag.service import AskResult, Source


class FakeService:
    def __init__(self):
        self.fail_generation = False

    def ask(self, question, history=None, language="en"):
        self.last_language = language
        if self.fail_generation:
            raise GenerationError("down")
        return AskResult(interaction_id=1, answer="resp [1]", rewritten_query="rw",
                         sources=[Source(doc_title="T", page=2, text="x", score=0.5)])

    def add_document(self, pdf_bytes, filename):
        if filename == "dup.pdf":
            raise DuplicateDocumentError("dup")
        if filename == "bad.pdf":
            raise ExtractionError("sem texto")
        return 7

    def add_documents(self, files):
        results = []
        for filename, data in files:
            try:
                added = self.add_document(data, filename)
                results.append({"filename": filename, "doc_id": Path(filename).stem,
                                "chunks_added": added, "error": None})
            except (DuplicateDocumentError, ExtractionError) as exc:
                results.append({"filename": filename, "doc_id": Path(filename).stem,
                                "chunks_added": 0, "error": str(exc)})
        return results

    def remove_document(self, doc_id):
        if doc_id == "missing":
            raise DocumentNotFoundError("not indexed")
        self.last_removed = doc_id
        return 2

    def reset_documents(self):
        return 5

    def restore_default_documents(self):
        if getattr(self, "fail_download", False):
            raise DownloadError("arxiv down")
        return {"documents_added": 5, "chunks_added": 42}

    def feedback(self, interaction_id, rating, comment=None):
        self.last_feedback = (interaction_id, rating, comment)

    def metrics(self):
        return {"total_questions": 0, "feedback_count": 0, "approval_rate": None,
                "approval_rate_7d": None, "avg_latency_ms": None,
                "negatives": [], "top_documents": []}

    def documents(self):
        return [{"doc_id": "d", "doc_title": "D", "chunks": 3}]

    def suggested_questions(self, language="en"):
        self.last_suggestions_language = language
        return ["Q1?", "Q2?", "Q3?"]


@pytest.fixture
def client():
    service = FakeService()
    app = create_app(service=service)
    return TestClient(app), service


def test_ask(client):
    c, _ = client
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "resp [1]"
    assert body["sources"][0]["doc_title"] == "T"


def test_ask_generation_down_returns_503(client):
    c, service = client
    service.fail_generation = True
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 503


def test_ask_passes_language(client):
    c, service = client
    c.post("/ask", json={"question": "q?", "language": "pt"})
    assert service.last_language == "pt"


def test_ask_defaults_to_english(client):
    c, service = client
    c.post("/ask", json={"question": "q?"})
    assert service.last_language == "en"


def test_ask_rejects_unknown_language(client):
    c, _ = client
    assert c.post("/ask", json={"question": "q?", "language": "fr"}).status_code == 422


def test_feedback_validates_rating(client):
    c, service = client
    ok = c.post("/feedback", json={"interaction_id": 1, "rating": 1})
    assert ok.status_code == 200
    assert service.last_feedback == (1, 1, None)
    bad = c.post("/feedback", json={"interaction_id": 1, "rating": 0})
    assert bad.status_code == 422


def _pdf_part(name):
    return ("files", (name, io.BytesIO(b"%PDF"), "application/pdf"))


def test_upload_single_file(client):
    c, _ = client
    r = c.post("/upload", files=[_pdf_part("ok.pdf")])
    assert r.status_code == 200
    results = r.json()["results"]
    assert results == [{"filename": "ok.pdf", "doc_id": "ok",
                        "chunks_added": 7, "error": None}]


def test_upload_multiple_files_reports_per_file_errors(client):
    c, _ = client
    r = c.post("/upload", files=[_pdf_part("ok.pdf"), _pdf_part("dup.pdf"),
                                 _pdf_part("bad.pdf")])
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["doc_id"] for x in results] == ["ok", "dup", "bad"]
    assert results[0]["error"] is None
    assert results[1]["error"] == "dup"
    assert results[2]["error"] == "sem texto"


def test_upload_batch_counts_once_against_rate_limit():
    service = FakeService()
    app = create_app(service=service, rate_limit=2, rate_window_s=60)
    c = TestClient(app)
    batch = [_pdf_part("a.pdf"), _pdf_part("b.pdf"), _pdf_part("c.pdf")]
    assert c.post("/upload", files=batch).status_code == 200
    assert c.post("/ask", json={"question": "q?"}).status_code == 200
    assert c.post("/ask", json={"question": "q?"}).status_code == 429


def test_delete_document(client):
    c, service = client
    r = c.delete("/documents/d")
    assert r.status_code == 200
    assert r.json() == {"doc_id": "d", "chunks_removed": 2}
    assert service.last_removed == "d"


def test_delete_missing_document_returns_404(client):
    c, _ = client
    assert c.delete("/documents/missing").status_code == 404


def test_reset_documents(client):
    c, _ = client
    r = c.post("/documents/reset")
    assert r.status_code == 200
    assert r.json() == {"chunks_removed": 5}


def test_restore_defaults(client):
    c, _ = client
    r = c.post("/documents/restore-defaults")
    assert r.status_code == 200
    assert r.json() == {"documents_added": 5, "chunks_added": 42}


def test_restore_defaults_download_failure_returns_502(client):
    c, service = client
    service.fail_download = True
    assert c.post("/documents/restore-defaults").status_code == 502


def test_rate_limit_shared_with_delete():
    service = FakeService()
    app = create_app(service=service, rate_limit=2, rate_window_s=60)
    c = TestClient(app)
    assert c.delete("/documents/d").status_code == 200
    assert c.post("/documents/reset").status_code == 200
    assert c.delete("/documents/d").status_code == 429


def test_health_documents_metrics(client):
    c, _ = client
    assert c.get("/health").json()["documents"] == 1
    assert c.get("/documents").json()[0]["doc_id"] == "d"
    assert "total_questions" in c.get("/metrics").json()


def test_question_too_long_rejected(client):
    c, _ = client
    assert c.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_question_at_limit_accepted(client):
    c, _ = client
    assert c.post("/ask", json={"question": "x" * 500}).status_code == 200


def test_rate_limit_returns_429():
    service = FakeService()
    app = create_app(service=service, rate_limit=2, rate_window_s=60)
    c = TestClient(app)
    assert c.post("/ask", json={"question": "q?"}).status_code == 200
    assert c.post("/ask", json={"question": "q?"}).status_code == 200
    r = c.post("/ask", json={"question": "q?"})
    assert r.status_code == 429
    assert "Rate limit" in r.json()["detail"]


def test_rate_limit_shared_with_upload():
    service = FakeService()
    app = create_app(service=service, rate_limit=2, rate_window_s=60)
    c = TestClient(app)
    assert c.post("/ask", json={"question": "q?"}).status_code == 200
    assert c.post("/upload", files=[_pdf_part("ok.pdf")]).status_code == 200
    assert c.post("/ask", json={"question": "q?"}).status_code == 429


def test_suggestions(client):
    c, service = client
    r = c.get("/suggestions", params={"language": "pt"})
    assert r.status_code == 200
    assert r.json() == {"questions": ["Q1?", "Q2?", "Q3?"]}
    assert service.last_suggestions_language == "pt"


def test_suggestions_defaults_to_english(client):
    c, service = client
    assert c.get("/suggestions").status_code == 200
    assert service.last_suggestions_language == "en"


def test_suggestions_rejects_unknown_language(client):
    c, _ = client
    assert c.get("/suggestions", params={"language": "fr"}).status_code == 422
