class ExtractionError(Exception):
    """PDF sem texto extraível ou corrompido."""


class GenerationError(Exception):
    """Falha ao chamar o LLM após esgotar as tentativas."""


class IndexNotFoundError(Exception):
    """Índice ausente em disco — rodar scripts/build_index.py."""


class DuplicateDocumentError(Exception):
    """Documento com mesmo doc_id já indexado."""
