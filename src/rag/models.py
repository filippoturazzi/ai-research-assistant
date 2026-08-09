from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    position: int
    text: str
