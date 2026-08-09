from rag.generation.generator import generate_answer
from rag.generation.groq_chat import GroqChat
from rag.generation.prompts import build_answer_messages, build_context
from rag.models import Chunk
from tests.fakes import FakeGroq


def _chunk(i, title, page, text):
    return Chunk(chunk_id=f"d:{i}", doc_id="d", doc_title=title, page=page, position=i, text=text)


def test_build_context_numbers_and_cites_sources():
    ctx = build_context([
        _chunk(0, "Attention Paper", 3, "Self-attention relates positions."),
        _chunk(1, "BERT Paper", 7, "Masked language modeling."),
    ])
    assert "[1] (Attention Paper, p. 3)" in ctx
    assert "[2] (BERT Paper, p. 7)" in ctx
    assert "Self-attention relates positions." in ctx


def test_messages_contain_rules_and_question():
    messages = build_answer_messages("O que é atenção?", [_chunk(0, "T", 1, "txt")])
    assert messages[0]["role"] == "system"
    assert "Não encontrei essa informação" in messages[0]["content"]
    assert "O que é atenção?" in messages[1]["content"]


def test_generate_answer_calls_model():
    fake = FakeGroq(["A atenção é... [1]"])
    out = generate_answer(GroqChat(client=fake), "O que é atenção?", [_chunk(0, "T", 1, "txt")])
    assert out == "A atenção é... [1]"
    assert fake.calls[0]["model"]
