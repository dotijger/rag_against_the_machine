class ImporterError(RAGError):
    pass

class ChunkError(RAGError):
    pass

class IndexingError(RAGError):
    pass

class RAGError(Exception):
    pass
