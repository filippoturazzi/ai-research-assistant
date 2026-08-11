import numpy as np
import pytest

from rag.errors import (DocumentNotFoundError, DuplicateDocumentError,
                        EmptyIndexError)
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
