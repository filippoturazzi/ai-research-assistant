from rag import config, errors


def test_config_constants():
    assert config.EMBEDDING_DIM == 384
    assert config.TOP_K == 5


def test_errors_are_exceptions():
    assert issubclass(errors.IndexNotFoundError, Exception)
    assert issubclass(errors.DuplicateDocumentError, Exception)
