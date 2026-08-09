import os
import time

from rag.errors import GenerationError

_MAX_ATTEMPTS = 3


class GroqChat:
    def __init__(self, api_key: str | None = None, client=None):
        if client is None:
            from groq import Groq  # import tardio
            client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self._client = client

    def complete(self, model: str, messages: list[dict],
                 max_tokens: int = 1024, temperature: float = 0.2) -> str:
        delay = 1.0
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise GenerationError(f"LLM indisponível: {exc}") from exc
                time.sleep(delay)
                delay *= 2
        raise GenerationError("unreachable")
