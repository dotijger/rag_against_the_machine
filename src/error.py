class RAGError(Exception):
    pass


class ImporterError(RAGError):
    pass


class ChunkError(RAGError):
    pass


class IndexerError(RAGError):
    pass
