
from src.rag import RAG, RAGError
from src.index import IndexError
from src.chunker import ChunkError

import sys
import fire


if __name__ == "__main__":
    try:
        fire.Fire(RAG())
    except (ValueError, RAGError, IndexError, ChunkError, TypeError) as e:
        print(e)
        sys.exit(1)
    except KeyboardInterrupt as e:
        print(e)
        sys.exit(1)
