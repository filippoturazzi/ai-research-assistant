from rag.models import Chunk

NO_ANSWER = "Não encontrei essa informação nos documentos."

_SYSTEM = f"""Você é um assistente de pesquisa que responde com base EXCLUSIVAMENTE \
no contexto fornecido (trechos de documentos numerados).

Regras:
1. Use apenas informações presentes no contexto. Não use conhecimento externo.
2. Cite a fonte de cada afirmação com o número entre colchetes, ex.: [1], [2].
3. Se o contexto não contém a resposta, diga exatamente: "{NO_ANSWER}"
4. Responda no idioma da pergunta."""


def build_context(chunks: list[Chunk]) -> str:
    blocks = [
        f"[{i}] ({c.doc_title}, p. {c.page})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def build_answer_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    user = f"Contexto:\n\n{build_context(chunks)}\n\nPergunta: {question}"
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
