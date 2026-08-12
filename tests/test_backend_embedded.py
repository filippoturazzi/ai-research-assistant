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

    def suggested_questions(self, language="en"):
        self.last_suggestions_language = language
        return ["Q1?", "Q2?", "Q3?"]


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


def test_embedded_document_management(embedded_backend, fake_service, monkeypatch):
    embedded_backend._hits.clear()
    monkeypatch.setattr(embedded_backend, "_session_cache", lambda: {})
    assert embedded_backend.remove_document("d") == {"doc_id": "d", "chunks_removed": 2}
    assert embedded_backend.reset_documents() == {"chunks_removed": 5}
    assert embedded_backend.restore_defaults() == {"documents_added": 1,
                                                   "chunks_added": 3}


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
    monkeypatch.setattr(embedded_backend, "_session_cache", lambda: {})
    monkeypatch.setattr(embedded_backend, "_new_session_service",
                        lambda version: calls.append(version) or "svc")
    assert embedded_backend._build_service() == "svc"
    assert calls == [embedded_backend._code_version()]


def test_build_service_purges_stale_rag_modules(embedded_backend, monkeypatch):
    import sys
    import types
    sys.modules["rag._stale_probe"] = types.ModuleType("rag._stale_probe")
    monkeypatch.setattr(sys, "_rag_loaded_version", "outdated", raising=False)
    monkeypatch.setattr(embedded_backend, "_session_cache", lambda: {})
    monkeypatch.setattr(embedded_backend, "_new_session_service", lambda version: "svc")

    embedded_backend._build_service()

    assert "rag._stale_probe" not in sys.modules
    assert sys._rag_loaded_version == embedded_backend._code_version()


def test_purge_also_clears_streamlit_resource_cache(embedded_backend, monkeypatch):
    import sys
    cleared = []
    monkeypatch.setattr(embedded_backend.st.cache_resource, "clear",
                        lambda: cleared.append(True), raising=False)
    monkeypatch.setattr(sys, "_rag_loaded_version", "outdated", raising=False)

    embedded_backend._ensure_fresh_rag()

    assert cleared


def test_no_cache_clear_when_version_current(embedded_backend, monkeypatch):
    import sys
    cleared = []
    monkeypatch.setattr(embedded_backend.st.cache_resource, "clear",
                        lambda: cleared.append(True), raising=False)
    monkeypatch.setattr(sys, "_rag_loaded_version",
                        embedded_backend._code_version(), raising=False)

    embedded_backend._ensure_fresh_rag()

    assert not cleared


def test_build_service_keeps_modules_when_version_current(embedded_backend, monkeypatch):
    import sys
    import types
    monkeypatch.setattr(sys, "_rag_loaded_version",
                        embedded_backend._code_version(), raising=False)
    sys.modules["rag._stale_probe"] = types.ModuleType("rag._stale_probe")
    monkeypatch.setattr(embedded_backend, "_session_cache", lambda: {})
    monkeypatch.setattr(embedded_backend, "_new_session_service", lambda version: "svc")

    embedded_backend._build_service()

    assert "rag._stale_probe" in sys.modules
    del sys.modules["rag._stale_probe"]


def test_embedded_service_build_failure_becomes_api_error(embedded_backend, monkeypatch):
    from rag.errors import GenerationError

    def boom():
        raise GenerationError("GROQ_API_KEY is not set")

    monkeypatch.setattr(embedded_backend, "_build_service", boom)
    with pytest.raises(embedded_backend.ApiError):
        embedded_backend.metrics()
    with pytest.raises(embedded_backend.ApiError):
        embedded_backend.documents()


def test_embedded_suggestions(embedded_backend, fake_service):
    assert embedded_backend.suggestions("pt") == ["Q1?", "Q2?", "Q3?"]
    assert fake_service.last_suggestions_language == "pt"


def test_embedded_suggestions_ignore_the_rate_limit(embedded_backend, fake_service,
                                                    monkeypatch):
    monkeypatch.setattr(embedded_backend, "_RATE_LIMIT", 1)
    embedded_backend._hits.clear()
    for _ in range(3):
        assert embedded_backend.suggestions("en") == ["Q1?", "Q2?", "Q3?"]


def test_http_mode_exports_suggestions(monkeypatch):
    monkeypatch.delenv("BACKEND_MODE", raising=False)
    import app.backend as backend
    importlib.reload(backend)
    import app.api_client as api_client
    assert backend.suggestions is api_client.suggestions


class _Nothing:
    """Stand-in for the shared, stateless resources — never called in these tests."""


@pytest.fixture
def session_backend(embedded_backend, monkeypatch, tmp_path):
    import numpy as np

    from rag.models import Chunk
    from rag.retrieval.store import IndexStore

    index_dir = tmp_path / "index"
    store = IndexStore(dim=4)
    store.add([Chunk(chunk_id="d:0", doc_id="d", doc_title="Doc D", page=1,
                     position=0, text="texto")], np.eye(4, dtype="float32")[:1])
    store.save(index_dir)
    monkeypatch.setattr("rag.config.INDEX_DIR", index_dir)
    monkeypatch.setattr(embedded_backend, "_shared_resources",
                        lambda code_version: {"embedder": _Nothing(),
                                              "reranker": _Nothing(),
                                              "chat": _Nothing(), "db": _Nothing()})
    return embedded_backend


def test_each_session_gets_its_own_store(session_backend):
    first = session_backend._new_session_service("v1")
    second = session_backend._new_session_service("v1")
    assert first.store is not second.store
    assert first.documents_dir != second.documents_dir
    assert [c.doc_id for c in first.store.chunks] == ["d"]
    assert [c.doc_id for c in second.store.chunks] == ["d"]


def test_the_session_service_is_ephemeral(session_backend):
    assert session_backend._new_session_service("v1").persist is False


def test_two_sessions_do_not_see_each_other(session_backend, monkeypatch):
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    tab_a, tab_b = {}, {}

    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab_a)
    service_a = session_backend._build_service()
    service_a.store.clear()

    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab_b)
    service_b = session_backend._build_service()

    assert service_a is not service_b
    assert service_a.store.chunks == []
    assert [c.doc_id for c in service_b.store.chunks] == ["d"]


def test_the_same_session_reuses_its_service(session_backend, monkeypatch):
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    tab = {}
    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab)
    assert session_backend._build_service() is session_backend._build_service()


def test_a_new_code_version_rebuilds_the_session_service(session_backend, monkeypatch):
    tab = {}
    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab)
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    first = session_backend._build_service()
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v2")
    assert session_backend._build_service() is not first


def test_restore_defaults_rebuilds_from_disk_without_downloading(session_backend,
                                                                monkeypatch):
    monkeypatch.setattr(session_backend, "_ensure_fresh_rag", lambda: "v1")
    tab = {}
    monkeypatch.setattr(session_backend, "_session_cache", lambda: tab)
    session_backend._hits.clear()
    session_backend._build_service().store.clear()

    def _no_download(*args, **kwargs):
        raise AssertionError("restore_defaults must not hit the network")

    monkeypatch.setattr("rag.ingestion.default_papers.fetch_default_papers",
                        _no_download)
    assert session_backend.restore_defaults() == {"documents_added": 1,
                                                  "chunks_added": 1}
    assert [c.doc_id for c in session_backend._build_service().store.chunks] == ["d"]
