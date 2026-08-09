from rag.config import REWRITE_MODEL
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat

_SYSTEMS = {
    "en": (
        "You rewrite the user's last question as a self-contained search query, "
        "resolving references to the conversation history and expanding acronyms "
        "when useful. The documents are in English, so write the search query in "
        "English. Reply ONLY with the rewritten query, without quotes or explanations."
    ),
    "pt": (
        "Você reescreve a última pergunta do usuário como uma consulta de busca "
        "autocontida, resolvendo referências ao histórico da conversa e expandindo "
        "siglas quando útil. Os documentos são em inglês, então escreva a consulta "
        "de busca em inglês. Responda APENAS com a consulta reescrita, sem aspas "
        "e sem explicações."
    ),
}


def rewrite_query(chat: GroqChat, question: str, history: list[dict],
                  language: str = "en") -> str:
    messages = (
        [{"role": "system", "content": _SYSTEMS[language]}]
        + history[-6:]
        + [{"role": "user", "content": f"Question: {question}\nSearch query:"}]
    )
    try:
        out = chat.complete(REWRITE_MODEL, messages, max_tokens=100)
    except GenerationError:
        return question
    out = out.strip().strip('"').strip()
    return out or question
