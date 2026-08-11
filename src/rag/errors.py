class ExtractionError(Exception):
    """PDF without extractable text, or corrupted."""


class GenerationError(Exception):
    """LLM call failed after exhausting retries."""


class IndexNotFoundError(Exception):
    """Index missing on disk — run scripts/build_index.py."""


class DuplicateDocumentError(Exception):
    """A document with the same doc_id is already indexed."""


class DocumentNotFoundError(Exception):
    """No indexed document with the given doc_id."""


class DownloadError(Exception):
    """Failed to download a default paper."""


class EmptyIndexError(GenerationError):
    """The knowledge base has no documents; nothing to answer from."""
