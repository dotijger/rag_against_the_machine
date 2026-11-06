import bm25s
from pathlib import Path
from pydantic import ValidationError
import json

from src.chunker import Chunk, Chunker
from src.index import Indexer
from src.models import (
        MinimalSource,
        MinimalSearchResults,
        UnansweredQuestion,
        AnsweredQuestion,
        MinimalAnswer,
        StudentSearchResults,
        StudentSearchResultsAndAnswer,
        RagDataset,
        )

class RAGError(BaseException):
    pass

class RAG():

    def __init__(self) -> None:
        self.indexer: None | Indexer = None
        self.chunked_corpus: list[Chunk] = []
        self.source_corpus: list[MinimalSource] = []
        self.output_path: None | Path = None
        self.rag_dataset: RagDataset(rag_questions=[])
        self.supported: list[str] = ['.md', '.py', '.txt']
   
    def _check_for_index(self) -> bool:
        if Path('data/processed').is_dir():
            with open('data/processed/chunks.jsonl', encoding='utf-8') as f:
                chunks = [Chunk.model_validate_json(line) for line in f]
            self.indexer = Indexer(corpus=chunks, loaded=True, output_path=Path('data/processed/'))
            self.chunked_corpus = chunks
            self.source_corpus = [self._to_minimal_source(c) for c in chunks]
            return True
        return False

    def _to_minimal_source(self, c: Chunk) -> MinimalSource:
        return MinimalSource(
                    file_path=str(c.file_path),
                    first_character_index=c.start,
                    last_character_index=c.end,
                )
    
    def _chunk(self, max_chunk_size: int, path: Path) -> None:
        for p in sorted(path.rglob('*')):
            if p.is_file() and p.suffix in self.supported:
                self._ingest_and_chunk(max_chunk_size, p)

    def _ingest_and_chunk(self, max_chunk_size: int, path: Path) -> None:
        chunker = Chunker(path=path, max_chunk_size=max_chunk_size)
        chunker.run()
        self.chunked_corpus.extend(chunker.chunks)
        minimal_sources = [self._to_minimal_source(c) for c in chunker.chunks]
        self.source_corpus.extend(minimal_sources)
    
    def index(self, max_chunk_size: int, raw_dir: str = 'data/raw/', out_dir: str = 'data/processed/') -> None:
        self.output_path = Path(out_dir)
        p_raw = Path(raw_dir)
        print(p_raw.resolve(), p_raw.exists())
        files = [p for p in p_raw.rglob("*") if p.is_file()]
        print(len(files), files[:5])
        self._chunk(max_chunk_size, Path(raw_dir))
        if self.chunked_corpus is not None:
            self.indexer = Indexer(self.chunked_corpus, False, self.output_path)
            self.indexer.index_corpus()
            print(f"Ingestion complete! Indexed {len(self.chunked_corpus)} chunks under {self.output_path}")
        else:
            raise RAGError("No corpus found to start indexing.")

    def _base_search(self, query: str, k: int) -> MinimalSearchResults:
        if not self._check_for_index():
            raise RAGError("Index must be built before other RAG function calls, aborting.")
        idx_results = self.indexer.get_ktop_idx(query, k)
        retrieved_sources = [self.source_corpus[i] for i in idx_results]
        return retrieved_sources

    def search(self, query: str, k: int, q_id: tuple[bool, UnansweredQuestion | None] = (False, None)) -> MinimalSearchResults:
        if not self._check_for_index():
            raise RAGError("Index must be built before other RAG function calls, aborting.")
        if not q_id[0]:
            uq = UnansweredQuestion(question=query)
        else:
            uq = q_id[1]
        print(f"{uq=}")
        result = MinimalSearchResults(
                question_id=uq.question_id,
                question=query,
                retrieved_sources=self._base_search(query, k),
                )
        if not q_id[0]:
            with open(f"search_results_{query}_{k}", "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2) + '\n')
        return result

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> StudentSearchResults:
        if not self._check_for_index():
            raise RAGError("Index must be built before other RAG function calls, aborting.")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f) # can raise JSON error 
        answers = []
        questions = dataset.get('rag_questions')
        if not questions:
            raise RAGError("Dataset does not contain any questions, aborting.")
        for question in questions:
            print(question)
            question_id = question.get('question_id')
            query = question.get('question')
            try:
                uq = UnansweredQuestion(
                        question_id=question_id,
                        question=query)
            except ValidationError as e:
                raise RAGError(f"Dataset question has an invalid format: {e}."
                               f"Attempted: {query=}, {question_id=}")
            answers.append(self.search(uq.question, k, (True, uq)))
        results = StudentSearchResults(search_results=answers, k=k)
        base = Path(dataset_path).name
        with open(save_directory / base, 'w', encoding='utf-8') as f:
            f.write(results.model_dump_json(indent=2) + '\n')
        return results

    def answer(self, query: str, k: int) -> MinimalAnswer:
        if self.indexer is None:
            raise RAGError("Index must be built before other RAG function calls, aborting.")
        # print(f"{query}, k = {k}")

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> StudentSearchResultsAndAnswer:
        if self.indexer is None:
            raise RAGError("Index must be built before other RAG function calls, aborting.")
        # print(f"{student_search_results_path=}, {save_directory=}")

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        if self.indexer is None:
            raise RAGError("Index must be built before other RAG function calls, aborting.")
        # print(f"{student_search_results_path=}, {dataset_path=}")
