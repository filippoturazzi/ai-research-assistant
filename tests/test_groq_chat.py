import pytest

from rag.errors import GenerationError
from rag.generation import groq_chat
from rag.generation.groq_chat import GroqChat
from tests.fakes import FakeGroq


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(groq_chat.time, "sleep", lambda s: None)


def test_returns_content():
    chat = GroqChat(client=FakeGroq(["hello"]))
    assert chat.complete("model-x", [{"role": "user", "content": "hi"}]) == "hello"


def test_retries_then_succeeds():
    fake = FakeGroq([RuntimeError("boom"), RuntimeError("boom"), "ok"])
    chat = GroqChat(client=fake)
    assert chat.complete("m", []) == "ok"
    assert len(fake.calls) == 3


def test_exhausted_raises_generation_error():
    fake = FakeGroq([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
    with pytest.raises(GenerationError):
        GroqChat(client=fake).complete("m", [])


def test_missing_api_key_raises_generation_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GenerationError):
        GroqChat()
