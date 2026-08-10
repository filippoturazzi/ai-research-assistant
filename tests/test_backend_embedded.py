import importlib

import pytest


@pytest.fixture
def embedded_backend(monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "embedded")
    import app.backend as backend
    importlib.reload(backend)
    yield backend
    monkeypatch.delenv("BACKEND_MODE", raising=False)
    importlib.reload(backend)


class FakeService:
    def __init__(self):
        self.fail = False

    def ask(self, question, history=None, language="en"):
        from rag.errors import GenerationError
        if self.fail:
            raise GenerationError("down")
        from rag.service import AskResult, Source
        return AskResult(interaction_id=7, answer="resp [1]", rewritten_query="rw",
                         sources=[Source(doc_title="T", page=2, text="x", score=0.5)])

    def add_document(self, data, filename):
        return 3

    def feedback(self, interaction_id, rating, comment=None):
        self.last = (interaction_id, rating)

    def metrics(self):
        return {"total_questions": 1}

    def documents(self):
        return [{"doc_id": "d", "doc_title": "D", "chunks": 3}]


@pytest.fixture
def fake_service(embedded_backend, monkeypatch):
    service = FakeService()
    monkeypatch.setattr(embedded_backend, "_build_service", lambda: service)
    return service


def test_http_is_default(monkeypatch):
    monkeypatch.delenv("BACKEND_MODE", raising=False)
    import app.backend as backend
    importlib.reload(backend)
    import app.api_client as api_client
    assert backend.ask is api_client.ask


def test_embedded_ask_returns_api_shaped_dict(embedded_backend, fake_service):
    out = embedded_backend.ask("q?", [], "en")
    assert out["interaction_id"] == 7
    assert out["answer"] == "resp [1]"
    assert out["sources"][0]["doc_title"] == "T"


def test_embedded_generation_error_becomes_api_error(embedded_backend, fake_service):
    fake_service.fail = True
    with pytest.raises(embedded_backend.ApiError):
        embedded_backend.ask("q?", [], "en")


def test_embedded_long_question_rejected(embedded_backend, fake_service):
    with pytest.raises(embedded_backend.ApiError, match="too long"):
        embedded_backend.ask("x" * 501, [], "en")


def test_embedded_rate_limit(embedded_backend, fake_service, monkeypatch):
    monkeypatch.setattr(embedded_backend, "_RATE_LIMIT", 2)
    embedded_backend._hits.clear()
    embedded_backend.ask("q?", [], "en")
    embedded_backend.upload("a.pdf", b"%PDF")
    with pytest.raises(embedded_backend.ApiError, match="Rate limit"):
        embedded_backend.ask("q?", [], "en")


def test_embedded_upload_and_misc(embedded_backend, fake_service):
    embedded_backend._hits.clear()
    out = embedded_backend.upload("novo_doc.pdf", b"%PDF")
    assert out == {"doc_id": "novo_doc", "chunks_added": 3}
    assert embedded_backend.send_feedback(7, 1) == {"ok": True}
    assert embedded_backend.metrics()["total_questions"] == 1
    assert embedded_backend.documents()[0]["doc_id"] == "d"
