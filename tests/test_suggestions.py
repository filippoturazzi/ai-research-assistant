from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat
from rag.generation.suggestions import suggest_questions
from rag.models import Chunk
from tests.fakes import FakeGroq


def _chunks(*docs):
    """Two chunks per document, positions 0 and 1."""
    out = []
    for doc_id, title in docs:
        for position in range(2):
            out.append(Chunk(chunk_id=f"{doc_id}:{position}", doc_id=doc_id,
                             doc_title=title, page=position + 1,
                             position=position, text=f"{title} texto {position}"))
    return out


def _broken_chat(monkeypatch):
    chat = GroqChat(client=FakeGroq([]))
    monkeypatch.setattr(chat, "complete",
                        lambda *a, **k: (_ for _ in ()).throw(GenerationError("down")))
    return chat


def test_returns_questions_from_the_llm():
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?"]))
    out = suggest_questions(chat, _chunks(("d", "Doc D")))
    assert out.questions == ["Q1?", "Q2?", "Q3?"]
    assert out.from_llm is True


def test_strips_numbering_bullets_and_quotes():
    chat = GroqChat(client=FakeGroq(['1. "Q1?"\n- Q2?\n3) Q3?\n\n']))
    assert suggest_questions(chat, _chunks(("d", "Doc D"))).questions == ["Q1?", "Q2?", "Q3?"]


def test_caps_at_n():
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?\nQ4?"]))
    assert suggest_questions(chat, _chunks(("d", "Doc D"))).questions == ["Q1?", "Q2?", "Q3?"]


def test_empty_base_returns_empty_without_calling_the_llm():
    fake = FakeGroq([])
    out = suggest_questions(GroqChat(client=fake), [])
    assert out.questions == []
    assert out.from_llm is False
    assert fake.calls == []


def test_generation_error_falls_back_to_titles(monkeypatch):
    out = suggest_questions(_broken_chat(monkeypatch),
                            _chunks(("a", "Paper A"), ("b", "Paper B")))
    assert len(out.questions) == 3
    assert all("Paper A" in q or "Paper B" in q for q in out.questions)
    assert out.from_llm is False


def test_short_output_falls_back():
    chat = GroqChat(client=FakeGroq(["Q1?\n\n"]))
    out = suggest_questions(chat, _chunks(("a", "Paper A")))
    assert len(out.questions) == 3
    assert all("Paper A" in q for q in out.questions)
    assert out.from_llm is False


def test_fallback_covers_n_with_a_single_document(monkeypatch):
    out = suggest_questions(_broken_chat(monkeypatch), _chunks(("a", "Paper A")))
    assert len(out.questions) == len(set(out.questions)) == 3


def test_prompt_samples_first_chunk_of_at_most_five_documents():
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?"]))
    docs = [(f"d{i}", f"Doc {i}") for i in range(7)]
    suggest_questions(chat, _chunks(*docs))
    user = chat._client.calls[0]["messages"][1]["content"]
    assert user.count("---") == 4      # 5 trechos -> 4 separadores
    assert "Doc 5" not in user         # sexto documento fora da amostra
    assert "texto 1" not in user       # só a posição 0 de cada documento


def test_uses_the_suggestion_model():
    from rag.config import SUGGESTION_MODEL
    chat = GroqChat(client=FakeGroq(["Q1?\nQ2?\nQ3?"]))
    suggest_questions(chat, _chunks(("d", "Doc D")))
    assert chat._client.calls[0]["model"] == SUGGESTION_MODEL


def test_portuguese_fallback_uses_portuguese_templates(monkeypatch):
    out = suggest_questions(_broken_chat(monkeypatch), _chunks(("a", "Paper A")),
                            language="pt")
    assert out.questions[0] == "O que o Paper A propõe?"


def test_preamble_line_is_rejected_and_falls_back():
    # A preamble like "Here are three questions:" doesn't end with "?", so it's
    # dropped by _parse. That leaves only 2 valid questions for n=3, so the
    # whole result falls back to the deterministic titles instead of letting
    # the preamble sneak in as a pill.
    chat = GroqChat(client=FakeGroq(["Here are three questions:\nQ1?\nQ2?"]))
    out = suggest_questions(chat, _chunks(("a", "Paper A")))
    assert out.from_llm is False
    assert len(out.questions) == 3
    assert all("Paper A" in q for q in out.questions)


def test_overlong_line_is_rejected_and_falls_back():
    long_question = "Q " + ("x" * 130) + "?"
    chat = GroqChat(client=FakeGroq([f"Q1?\n{long_question}\nQ3?"]))
    out = suggest_questions(chat, _chunks(("a", "Paper A")))
    assert out.from_llm is False
    assert len(out.questions) == 3
