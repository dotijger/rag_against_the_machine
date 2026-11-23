class RAGError(Exception):
    pass


class EmptyQueryError(RAGError):
    pass


class ImporterError(RAGError):
    pass


class ChunkError(RAGError):
    pass


class IndexerError(RAGError):
    pass
