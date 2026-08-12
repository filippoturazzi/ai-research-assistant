import re

from rag.config import SUGGESTION_MODEL
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat
from rag.models import Chunk

_MAX_DOCS = 5
_EXCERPT_CHARS = 600

_SYSTEMS = {
    "en": (
        "You write example questions for a research assistant's knowledge base. "
        "From the excerpts of the indexed documents, write exactly {n} short "
        "questions those documents can answer. Each question must be "
        "self-contained, under 80 characters, and about the content of the "
        "documents. Reply with one question per line and nothing else: no "
        "numbering, no bullets, no quotes, no commentary."
    ),
    "pt": (
        "Você escreve perguntas de exemplo para a base de conhecimento de um "
        "assistente de pesquisa. A partir dos trechos dos documentos indexados, "
        "escreva exatamente {n} perguntas curtas que esses documentos consigam "
        "responder. Cada pergunta deve ser autocontida, ter menos de 80 "
        "caracteres e falar do conteúdo dos documentos. Responda com uma "
        "pergunta por linha e nada mais: sem numeração, sem marcadores, sem "
        "aspas e sem comentários."
    ),
}

_FALLBACKS = {
    "en": ["What does {title} propose?",
           "What are the main findings in {title}?",
           "How does {title} evaluate its approach?"],
    "pt": ["O que o {title} propõe?",
           "Quais são os principais resultados de {title}?",
           "Como {title} avalia a abordagem?"],
}


def _sample(chunks: list[Chunk]) -> list[Chunk]:
    # One excerpt per document — the lowest position is the start of the paper,
    # where the abstract and the introduction live. Sorting by doc_id keeps the
    # prompt identical across calls on an unchanged knowledge base.
    first: dict[str, Chunk] = {}
    for chunk in chunks:
        current = first.get(chunk.doc_id)
        if current is None or chunk.position < current.position:
            first[chunk.doc_id] = chunk
    return [first[doc_id] for doc_id in sorted(first)][:_MAX_DOCS]


def _clean(line: str) -> str:
    line = re.sub(r"^\s*[-*•]\s*", "", line.strip())
    line = re.sub(r"^\d+[.)]\s*", "", line)
    return line.strip().strip('"').strip("'").strip()


def _parse(raw: str, n: int) -> list[str]:
    questions = [_clean(line) for line in raw.splitlines()]
    return [q for q in questions if q][:n]


def _fallback(sample: list[Chunk], language: str, n: int) -> list[str]:
    titles = [c.doc_title for c in sample]
    templates = _FALLBACKS[language]
    return [templates[i % len(templates)].format(title=titles[i % len(titles)])
            for i in range(n)]


def suggest_questions(chat: GroqChat, chunks: list[Chunk],
                      language: str = "en", n: int = 3) -> list[str]:
    sample = _sample(chunks)
    if not sample:
        return []
    excerpts = "\n\n---\n\n".join(
        f"({c.doc_title})\n{c.text[:_EXCERPT_CHARS]}" for c in sample)
    messages = [
        {"role": "system", "content": _SYSTEMS[language].format(n=n)},
        {"role": "user", "content": f"Excerpts:\n\n{excerpts}"},
    ]
    try:
        raw = chat.complete(SUGGESTION_MODEL, messages, max_tokens=200,
                            temperature=0.5)
    except GenerationError:
        return _fallback(sample, language, n)
    questions = _parse(raw, n)
    return questions if len(questions) == n else _fallback(sample, language, n)
