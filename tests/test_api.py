import io

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from rag.errors import DuplicateDocumentError, ExtractionError, GenerationError
from rag.service import AskResult, Source


class FakeService:
    def __init__(self):
        self.fail_generation = False

    def ask(self, question, history=None):
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

    def feedback(self, interaction_id, rating, comment=None):
        self.last_feedback = (interaction_id, rating, comment)

    def metrics(self):
        return {"total_questions": 0, "feedback_count": 0, "approval_rate": None,
                "approval_rate_7d": None, "avg_latency_ms": None,
                "negatives": [], "top_documents": []}

    def documents(self):
        return [{"doc_id": "d", "doc_title": "D", "chunks": 3}]


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


def test_feedback_validates_rating(client):
    c, service = client
    ok = c.post("/feedback", json={"interaction_id": 1, "rating": 1})
    assert ok.status_code == 200
    assert service.last_feedback == (1, 1, None)
    bad = c.post("/feedback", json={"interaction_id": 1, "rating": 0})
    assert bad.status_code == 422


def test_upload_paths(client):
    c, _ = client
    pdf = ("file", ("ok.pdf", io.BytesIO(b"%PDF"), "application/pdf"))
    assert c.post("/upload", files=[pdf]).status_code == 200
    dup = ("file", ("dup.pdf", io.BytesIO(b"%PDF"), "application/pdf"))
    assert c.post("/upload", files=[dup]).status_code == 409
    bad = ("file", ("bad.pdf", io.BytesIO(b"%PDF"), "application/pdf"))
    assert c.post("/upload", files=[bad]).status_code == 422


def test_health_documents_metrics(client):
    c, _ = client
    assert c.get("/health").json()["documents"] == 1
    assert c.get("/documents").json()[0]["doc_id"] == "d"
    assert "total_questions" in c.get("/metrics").json()
