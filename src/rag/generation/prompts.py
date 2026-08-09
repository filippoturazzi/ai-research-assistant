from rag.models import Chunk

NO_ANSWER = {
    "en": "I could not find this information in the documents.",
    "pt": "Não encontrei essa informação nos documentos.",
}

_SYSTEMS = {
    "en": f"""You are a research assistant that answers based EXCLUSIVELY on the \
provided context (numbered document excerpts).

Rules:
1. Use only information present in the context. Do not use outside knowledge.
2. Cite the source of every claim with its number in brackets, e.g. [1], [2].
3. If the context does not contain the answer, say exactly: "{NO_ANSWER['en']}"
4. Answer in English.""",
    "pt": f"""Você é um assistente de pesquisa que responde com base EXCLUSIVAMENTE \
no contexto fornecido (trechos de documentos numerados).

Regras:
1. Use apenas informações presentes no contexto. Não use conhecimento externo.
2. Cite a fonte de cada afirmação com o número entre colchetes, ex.: [1], [2].
3. Se o contexto não contém a resposta, diga exatamente: "{NO_ANSWER['pt']}"
4. Responda em português.""",
}


def build_context(chunks: list[Chunk]) -> str:
    blocks = [
        f"[{i}] ({c.doc_title}, p. {c.page})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


def build_answer_messages(question: str, chunks: list[Chunk],
                          language: str = "en") -> list[dict]:
    user = f"Context:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": _SYSTEMS[language]},
        {"role": "user", "content": user},
    ]
