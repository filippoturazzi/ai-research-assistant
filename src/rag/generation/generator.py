from rag.config import GENERATION_MODEL
from rag.generation.groq_chat import GroqChat
from rag.generation.prompts import build_answer_messages
from rag.models import Chunk


def generate_answer(chat: GroqChat, question: str, chunks: list[Chunk],
                    language: str = "en") -> str:
    return chat.complete(GENERATION_MODEL, build_answer_messages(question, chunks, language))
