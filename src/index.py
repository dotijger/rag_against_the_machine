from pathlib import Path
from pydantic import BaseModel, model_validator
import bm25s
from tqdm import tqdm
import json


from src.chunker import Chunker, Chunk

class Indexer():
    
    def __init__(self, corpus: list[Chunk], loaded: bool, output_path: Path) -> None:
        self.corpus = corpus
        if not loaded:
            self.retriever = bm25s.BM25()
        else:
            self.retriever = bm25s.BM25.load(output_path / 'bm25s')
        self.output_path = output_path

    def _save_chunk_lib(self) -> None:
        with open(self.output_path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.corpus:
                f.write(c.model_dump_json() + '\n')

    def index_corpus(self) -> None:
        corpus_tokens = bm25s.tokenize([c.text for c in self.corpus], show_progress=True)
        print(len(self.corpus), sum(len(t) for t in corpus_tokens))
        self.retriever.index(corpus_tokens, show_progress=True)
        self.retriever.save(self.output_path / "bm25s")
        self._save_chunk_lib()

    def get_ktop_idx(self, query: str, k: int) -> list[int]:
        ids, scores = self.retriever.retrieve(bm25s.tokenize(query), k=k)
        return [int(i) for i in ids[0]]
        

