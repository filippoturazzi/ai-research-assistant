import numpy as np
import pytest

from rag.errors import (DocumentNotFoundError, DuplicateDocumentError,
                        EmptyIndexError, ExtractionError, GenerationError)
from rag.feedback.db import FeedbackDB
from rag.models import Chunk
from rag.retrieval.store import IndexStore
from rag.service import AskResult, RAGService
from rag.generation.groq_chat import GroqChat
from tests.fakes import FakeGroq


class FakeEmbedder:
    def embed_query(self, text):
        return np.eye(4, dtype="float32")[0]

    def embed_texts(self, texts):
        return np.ones((len(texts), 4), dtype="float32")


class FakeReranker:
    def rerank(self, query, chunks, top_k):
        return [(c, 1.0) for c in chunks[:top_k]]


@pytest.fixture
def service(tmp_path):
    store = IndexStore(dim=4)
    chunks = [Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title="Doc D", page=i + 1,
                    position=i, text=f"chunk text {i}") for i in range(3)]
    store.add(chunks, np.eye(4, dtype="float32")[:3])
    chat = GroqChat(client=FakeGroq(["query reescrita", "resposta final [1]"]))
    db = FeedbackDB(":memory:")
    svc = RAGService(store=store, embedder=FakeEmbedder(), reranker=FakeReranker(),
                     chat=chat, db=db, index_dir=tmp_path / "index",
                     documents_dir=tmp_path / "docs")
    yield svc
    db.close()


def test_ask_returns_answer_with_sources_and_logs(service):
    result = service.ask("qual é o chunk?")
    assert isinstance(result, AskResult)
    assert result.answer == "resposta final [1]"
    assert result.rewritten_query == "query reescrita"
    assert result.sources and result.sources[0].doc_title == "Doc D"
    assert service.metrics()["total_questions"] == 1


def test_feedback_links_to_interaction(service):
    result = service.ask("pergunta")
    service.feedback(result.interaction_id, 1)
    assert service.metrics()["approval_rate"] == 1.0


def test_documents_lists_indexed(service):
    docs = service.documents()
    assert docs == [{"doc_id": "d", "doc_title": "Doc D", "chunks": 3}]


def test_add_document_persists_index(service, sample_pdf, tmp_path):
    added = service.add_document(sample_pdf.read_bytes(), "novo_paper.pdf")
    assert added > 0
    assert (tmp_path / "index" / "chunks.json").exists()
    assert (tmp_path / "docs" / "novo_paper.pdf").exists()
    assert any(d["doc_id"] == "novo_paper" for d in service.documents())


def test_add_document_sanitizes_filename(service, sample_pdf, tmp_path):
    added = service.add_document(sample_pdf.read_bytes(), "../evil.pdf")
    assert added > 0
    assert (tmp_path / "docs" / "evil.pdf").exists()
    assert not (tmp_path / "evil.pdf").exists()


def test_add_document_duplicate_does_not_overwrite_pdf(service, sample_pdf, tmp_path):
    service.add_document(sample_pdf.read_bytes(), "dup_doc.pdf")
    stored_path = tmp_path / "docs" / "dup_doc.pdf"
    original_bytes = stored_path.read_bytes()

    with pytest.raises(DuplicateDocumentError):
        service.add_document(b"other bytes", "dup_doc.pdf")

    assert stored_path.read_bytes() == original_bytes


def test_add_documents_batch_reports_per_file_results(service, sample_pdf, tmp_path):
    data = sample_pdf.read_bytes()

    results = service.add_documents([
        ("paper_um.pdf", data),
        ("paper_um.pdf", data),   # duplicate within the batch
        ("paper_dois.pdf", data),
    ])

    assert [r["doc_id"] for r in results] == ["paper_um", "paper_um", "paper_dois"]
    assert results[0]["chunks_added"] > 0 and results[0]["error"] is None
    assert results[1]["chunks_added"] == 0 and "paper_um" in results[1]["error"]
    assert results[2]["chunks_added"] > 0 and results[2]["error"] is None
    doc_ids = {d["doc_id"] for d in service.documents()}
    assert {"paper_um", "paper_dois"} <= doc_ids
    assert (tmp_path / "docs" / "paper_dois.pdf").exists()


def test_remove_document_deletes_pdf_and_persists(service, sample_pdf, tmp_path):
    service.add_document(sample_pdf.read_bytes(), "novo_paper.pdf")

    removed = service.remove_document("novo_paper")

    assert removed > 0
    assert not (tmp_path / "docs" / "novo_paper.pdf").exists()
    assert all(d["doc_id"] != "novo_paper" for d in service.documents())
    chunks_json = (tmp_path / "index" / "chunks.json").read_text(encoding="utf-8")
    assert "novo_paper" not in chunks_json


def test_remove_document_missing_raises(service):
    with pytest.raises(DocumentNotFoundError):
        service.remove_document("nao_existe")


def test_reset_documents_empties_collection(service, sample_pdf, tmp_path):
    service.add_document(sample_pdf.read_bytes(), "novo_paper.pdf")

    removed = service.reset_documents()

    assert removed > 0
    assert service.documents() == []
    assert list((tmp_path / "docs").glob("*.pdf")) == []
    assert (tmp_path / "index" / "chunks.json").exists()


def test_restore_default_documents_replaces_collection(service, sample_pdf):
    def fake_fetch():
        yield "paper_um.pdf", sample_pdf.read_bytes()
        yield "paper_dois.pdf", sample_pdf.read_bytes()

    result = service.restore_default_documents(fetch=fake_fetch)

    assert result["documents_added"] == 2
    assert result["chunks_added"] > 0
    doc_ids = {d["doc_id"] for d in service.documents()}
    assert doc_ids == {"paper_um", "paper_dois"}  # old doc "d" gone


def test_ask_empty_base_raises_friendly_error(service):
    service.reset_documents()
    with pytest.raises(EmptyIndexError):
        service.ask("pergunta?")


def test_ask_portuguese_uses_pt_prompt(service):
    result = service.ask("qual é o chunk?", language="pt")
    assert result.answer == "resposta final [1]"
    # the second Groq call is the generation; its system prompt must be the PT one
    generation_call = service.chat._client.calls[1]
    assert "Responda em português." in generation_call["messages"][0]["content"]


@pytest.fixture
def suggest_service(tmp_path):
    store = IndexStore(dim=4)
    chunks = [Chunk(chunk_id=f"{doc}:0", doc_id=doc, doc_title=f"Paper {doc.upper()}",
                    page=1, position=0, text=f"texto do {doc}")
              for doc in ("a", "b")]
    store.add(chunks, np.eye(4, dtype="float32")[:2])
    fake = FakeGroq(["Q1?\nQ2?\nQ3?", "R1?\nR2?\nR3?"])
    db = FeedbackDB(":memory:")
    svc = RAGService(store=store, embedder=FakeEmbedder(), reranker=FakeReranker(),
                     chat=GroqChat(client=fake), db=db,
                     index_dir=tmp_path / "index", documents_dir=tmp_path / "docs")
    yield svc, fake
    db.close()


def test_suggested_questions_are_cached_for_an_unchanged_base(suggest_service):
    svc, fake = suggest_service
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]
    assert len(fake.calls) == 1


def test_suggested_questions_regenerate_after_the_base_changes(suggest_service):
    svc, fake = suggest_service
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]
    svc.remove_document("b")
    assert svc.suggested_questions("en") == ["R1?", "R2?", "R3?"]
    assert len(fake.calls) == 2


def test_suggested_questions_are_cached_per_language(suggest_service):
    svc, fake = suggest_service
    svc.suggested_questions("en")
    svc.suggested_questions("pt")
    svc.suggested_questions("en")
    assert len(fake.calls) == 2


def test_suggested_questions_on_an_empty_base(tmp_path):
    fake = FakeGroq([])
    db = FeedbackDB(":memory:")
    svc = RAGService(store=IndexStore(dim=4), embedder=FakeEmbedder(),
                     reranker=FakeReranker(), chat=GroqChat(client=fake), db=db,
                     index_dir=tmp_path / "index", documents_dir=tmp_path / "docs")
    assert svc.suggested_questions("en") == []
    assert fake.calls == []
    db.close()


def test_suggested_questions_fallback_result_is_not_cached(tmp_path, monkeypatch):
    # A fallback (e.g. from a Groq 429 on the first call) must not get pinned
    # in the cache in place of real suggestions — the next call should retry
    # the LLM. An LLM-backed result, by contrast, IS cached (already covered
    # by test_suggested_questions_are_cached_for_an_unchanged_base above).
    store = IndexStore(dim=4)
    chunks = [Chunk(chunk_id="a:0", doc_id="a", doc_title="Paper A", page=1,
                    position=0, text="texto do a")]
    store.add(chunks, np.eye(4, dtype="float32")[:1])
    chat = GroqChat(client=FakeGroq([]))
    db = FeedbackDB(":memory:")
    svc = RAGService(store=store, embedder=FakeEmbedder(), reranker=FakeReranker(),
                     chat=chat, db=db, index_dir=tmp_path / "index",
                     documents_dir=tmp_path / "docs")

    calls = {"n": 0}

    def flaky_complete(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise GenerationError("down")
        return "Q1?\nQ2?\nQ3?"

    monkeypatch.setattr(chat, "complete", flaky_complete)

    first = svc.suggested_questions("en")
    assert len(first) == 3 and "Paper A" in first[0]  # deterministic fallback

    second = svc.suggested_questions("en")
    assert second == ["Q1?", "Q2?", "Q3?"]
    assert calls["n"] == 2  # both calls reached the LLM — the fallback wasn't cached
    db.close()


def test_add_document_invalidates_suggestions_cache(suggest_service, sample_pdf):
    svc, fake = suggest_service
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]

    svc.add_document(sample_pdf.read_bytes(), "novo_paper.pdf")

    assert svc.suggested_questions("en") == ["R1?", "R2?", "R3?"]
    assert len(fake.calls) == 2


def test_reset_documents_invalidates_suggestions_cache(suggest_service):
    svc, fake = suggest_service
    assert svc.suggested_questions("en") == ["Q1?", "Q2?", "Q3?"]

    svc.reset_documents()

    # An emptied base has nothing to sample, so suggestions come back empty
    # without a second LLM call, and that empty result is itself cached.
    assert svc.suggested_questions("en") == []
    assert svc.suggested_questions("en") == []
    assert len(fake.calls) == 1


def _persist_service(tmp_path, persist):
    store = IndexStore(dim=4)
    db = FeedbackDB(":memory:")
    svc = RAGService(store=store, embedder=FakeEmbedder(), reranker=FakeReranker(),
                     chat=GroqChat(client=FakeGroq([])), db=db,
                     index_dir=tmp_path / "index",
                     documents_dir=tmp_path / "docs", persist=persist)
    return svc, db


def test_ephemeral_add_document_never_writes_the_index(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=False)
    assert svc.add_document(sample_pdf.read_bytes(), "a.pdf") > 0
    assert not (tmp_path / "index").exists()
    db.close()


def test_ephemeral_add_document_deletes_the_pdf_after_indexing(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=False)
    svc.add_document(sample_pdf.read_bytes(), "a.pdf")
    assert list((tmp_path / "docs").glob("*.pdf")) == []
    db.close()


def test_ephemeral_add_document_deletes_the_pdf_even_when_ingestion_fails(tmp_path):
    svc, db = _persist_service(tmp_path, persist=False)
    with pytest.raises(ExtractionError):
        svc.add_document(b"not a pdf at all", "bad.pdf")
    assert list((tmp_path / "docs").glob("*.pdf")) == []
    db.close()


def test_ephemeral_remove_and_reset_never_write_the_index(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=False)
    svc.add_document(sample_pdf.read_bytes(), "a.pdf")
    svc.add_document(sample_pdf.read_bytes(), "b.pdf")
    svc.remove_document("a")
    svc.reset_documents()
    assert not (tmp_path / "index").exists()
    db.close()


def test_persisting_service_still_writes_the_index_and_keeps_the_pdf(tmp_path, sample_pdf):
    svc, db = _persist_service(tmp_path, persist=True)
    svc.add_document(sample_pdf.read_bytes(), "a.pdf")
    assert (tmp_path / "index" / "chunks.json").exists()
    assert (tmp_path / "docs" / "a.pdf").exists()
    db.close()
