
from src.rag import RAG
from src.error import (
        RAGError,
        IndexError,
        ChunkError,
        ImporterError,
        )

import sys
import fire


if __name__ == "__main__":
    try:
        fire.Fire(RAG())
    except (ValueError, RAGError, IndexError, ChunkError, ImporterError, TypeError) as e:
        print(e)
        sys.exit(1)
    except KeyboardInterrupt as e:
        print(e)
        sys.exit(1)
