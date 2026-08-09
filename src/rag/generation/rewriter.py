from rag.config import REWRITE_MODEL
from rag.errors import GenerationError
from rag.generation.groq_chat import GroqChat

_SYSTEM = (
    "Você reescreve a última pergunta do usuário como uma consulta de busca "
    "autocontida, resolvendo referências ao histórico da conversa e expandindo "
    "siglas quando útil. Responda APENAS com a consulta reescrita, sem aspas "
    "e sem explicações."
)


def rewrite_query(chat: GroqChat, question: str, history: list[dict]) -> str:
    messages = (
        [{"role": "system", "content": _SYSTEM}]
        + history[-6:]
        + [{"role": "user", "content": f"Pergunta: {question}\nConsulta de busca:"}]
    )
    try:
        out = chat.complete(REWRITE_MODEL, messages, max_tokens=100)
    except GenerationError:
        return question
    out = out.strip().strip('"').strip()
    return out or question
