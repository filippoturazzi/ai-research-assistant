from typing import Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(max_length=500)
    history: list[HistoryMessage] = []
    language: Literal["en", "pt"] = "en"


class SourceOut(BaseModel):
    doc_title: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    interaction_id: int
    answer: str
    rewritten_query: str
    sources: list[SourceOut]


class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: Literal[1, -1]
    comment: str | None = None


class UploadResponse(BaseModel):
    doc_id: str
    chunks_added: int
