class RAGError(Exception):
    """Base class for all expected errors in the RAG pipeline."""

    pass


class EmptyQueryError(RAGError):
    """A query has no searchable terms left after tokenization."""

    pass


class ImporterError(RAGError):
    """A file could not be read, parsed or validated against its model."""

    pass


class ChunkError(RAGError):
    """A chunk failed validation while a file was being chunked."""

    pass


class IndexerError(RAGError):
    """The index is missing, out of date, or was given an unknown mode."""

    pass
