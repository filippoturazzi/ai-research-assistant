from rag.config import CHUNK_WORDS, OVERLAP_WORDS
from rag.models import Chunk


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    doc_title: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    # units (page, paragraph); paragraphs larger than the limit are sliced
    units: list[tuple[int, str]] = []
    for page, text in pages:
        for para in _split_paragraphs(text):
            words = para.split()
            if len(words) <= chunk_words:
                units.append((page, para))
            else:
                for i in range(0, len(words), chunk_words):
                    units.append((page, " ".join(words[i:i + chunk_words])))

    chunks: list[Chunk] = []
    current: list[tuple[int, str]] = []
    current_words = 0
    overlap_text = ""

    def emit() -> None:
        nonlocal current, current_words, overlap_text
        if not current:
            return
        page = current[0][0]
        body = "\n\n".join(p for _, p in current)
        text = f"{overlap_text}\n\n{body}" if overlap_text else body
        position = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}:{position}", doc_id=doc_id, doc_title=doc_title,
            page=page, position=position, text=text,
        ))
        overlap_text = " ".join(body.split()[-overlap_words:])
        current, current_words = [], 0

    for page, para in units:
        n = len(para.split())
        if current and current_words + n > chunk_words:
            emit()
        current.append((page, para))
        current_words += n
    emit()
    return chunks
