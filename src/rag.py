import bm25s
from pathlib import Path

from src.chunker import Chunk, Chunker
from src.index import Indexer
from src.models import MinimalSource


class RAG():
    self.indexer: None | Indexer = None
    self.path: Path = Path.cwd()
    self.chunked_corpus: None | List[Chunk] = None
    self.source_corpus: None | List[MinimalSource] = None
    self.output_path: Path = Path('data/processed/')

    def _ingest_and_chunk(self) -> None:
        chunker = Chunker(path=self.path, max_chunk_size=max_chunk_size)
        chunker.run()
        self.chunked_corpus = chunker.chunks
        self.source_corpus = [
                MinimalSource(
                    file_path=c.file_path,
                    first_character_index=c.start,
                    last_character_index=c.end,
                    ) for c in self.chunked_corpus
                ]

    def index(max_chunk_size: int) -> None:
        print(f"{max_chunk_size=}")
        self._ingest_and_chunk(self, max_chunk_size)
        self.indexer = Indexer(self.corpus, self.output_path)
        self.indexer.index_corpus()
        print(f"Ingestion complete! Indexed {len(self.chunked_corpus)} chunks under {self.output_path}")

    def search(query: str, k: int) -> None:
        print(f"{query}, k = {k}")
        idx_results = self.indexer.get_ktop_idx(query, k)
        for i in idx_results:
            source = self.source_corpus[i]
            print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")


    def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
        print(f"{dataset_path=}, {k=}, {save_directory=}")

    def answer(query: str, k: int) -> None:
        print(f"{query}, k = {k}")

    def answer_dataset(student_search_results_path: str, save_directory: str) -> None:
        print(f"{student_search_results_path=}, {save_directory=}")

    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        print(f"{student_search_results_path=}, {dataset_path=}")
