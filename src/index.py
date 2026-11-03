from pathlib import Path
from pydantic import BaseModel, model_validator
import bm25s


from src.chunker import Chunker, Chunk


class IndexError(BaseException):
    pass

class Indexer():
    
    def __init__(self, corpus: list[Chunk], output_path: Path) -> None:
        self.corpus = corpus
        self.retriever = bm25s.BM25()
        self.output_path = output_path

    def index_corpus(self) -> None:
        retriever.index([bm25s.tokenize(chunk.text) for chunk in corpus])

    def get_ktop_idx(self, query: str, k: int) -> list[int]:
        ids, scores = self.retriever.retrieve([bm25s.tokenize(query)], k=k)
        return [int(i) for i in ids[0]]
        

