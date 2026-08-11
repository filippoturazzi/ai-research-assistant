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

    def add_documents(self, files):
        from pathlib import Path
        return [{"filename": f, "doc_id": Path(f).stem, "chunks_added": 3,
                 "error": None} for f, _ in files]

    def remove_document(self, doc_id):
        from rag.errors import DocumentNotFoundError
        if doc_id == "missing":
            raise DocumentNotFoundError("not indexed")
        return 2

    def reset_documents(self):
        return 5

    def restore_default_documents(self):
        return {"documents_added": 5, "chunks_added": 42}

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
    embedded_backend.upload([("a.pdf", b"%PDF"), ("b.pdf", b"%PDF")])
    with pytest.raises(embedded_backend.ApiError, match="Rate limit"):
        embedded_backend.ask("q?", [], "en")


def test_embedded_upload_and_misc(embedded_backend, fake_service):
    embedded_backend._hits.clear()
    out = embedded_backend.upload([("novo_doc.pdf", b"%PDF"), ("outro.pdf", b"%PDF")])
    assert [r["doc_id"] for r in out["results"]] == ["novo_doc", "outro"]
    assert out["results"][0]["chunks_added"] == 3
    assert embedded_backend.send_feedback(7, 1) == {"ok": True}
    assert embedded_backend.metrics()["total_questions"] == 1
    assert embedded_backend.documents()[0]["doc_id"] == "d"


def test_embedded_document_management(embedded_backend, fake_service):
    embedded_backend._hits.clear()
    assert embedded_backend.remove_document("d") == {"doc_id": "d", "chunks_removed": 2}
    assert embedded_backend.reset_documents() == {"chunks_removed": 5}
    assert embedded_backend.restore_defaults() == {"documents_added": 5,
                                                   "chunks_added": 42}


def test_embedded_remove_missing_becomes_api_error(embedded_backend, fake_service):
    embedded_backend._hits.clear()
    with pytest.raises(embedded_backend.ApiError):
        embedded_backend.remove_document("missing")


def test_code_version_stable_until_source_changes(embedded_backend, tmp_path):
    (tmp_path / "a.py").write_text("x = 1")
    v1 = embedded_backend._code_version(tmp_path)
    assert v1 == embedded_backend._code_version(tmp_path)
    (tmp_path / "a.py").write_text("x = 2")
    assert embedded_backend._code_version(tmp_path) != v1


def test_build_service_keys_cache_on_current_code_version(embedded_backend, monkeypatch):
    calls = []
    monkeypatch.setattr(embedded_backend, "_cached_service",
                        lambda version: calls.append(version) or "svc")
    assert embedded_backend._build_service() == "svc"
    assert calls == [embedded_backend._code_version()]


def test_embedded_service_build_failure_becomes_api_error(embedded_backend, monkeypatch):
    from rag.errors import GenerationError

    def boom():
        raise GenerationError("GROQ_API_KEY is not set")

    monkeypatch.setattr(embedded_backend, "_build_service", boom)
    with pytest.raises(embedded_backend.ApiError):
        embedded_backend.metrics()
    with pytest.raises(embedded_backend.ApiError):
        embedded_backend.documents()
