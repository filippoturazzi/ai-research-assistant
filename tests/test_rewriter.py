from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat
from rag.generation.rewriter import rewrite_query
from tests.fakes import FakeGroq


def test_returns_rewritten_query():
    chat = GroqChat(client=FakeGroq(['"limitações da arquitetura Transformer"']))
    out = rewrite_query(chat, "e as limitações disso?",
                        [{"role": "user", "content": "o que é o Transformer?"}])
    assert out == "limitações da arquitetura Transformer"


def test_falls_back_to_original_on_error(monkeypatch):
    chat = GroqChat(client=FakeGroq([]))
    monkeypatch.setattr(chat, "complete",
                        lambda *a, **k: (_ for _ in ()).throw(GenerationError("down")))
    assert rewrite_query(chat, "pergunta original", []) == "pergunta original"


def test_falls_back_on_empty_output():
    chat = GroqChat(client=FakeGroq(["   "]))
    assert rewrite_query(chat, "pergunta original", []) == "pergunta original"


def test_rewriter_portuguese_template_still_targets_english_query():
    from rag.generation import rewriter
    assert "search query in English" in rewriter._SYSTEMS["en"]
    assert "consulta de busca em inglês" in rewriter._SYSTEMS["pt"]
