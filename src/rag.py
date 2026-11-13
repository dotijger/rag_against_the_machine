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
        Evaluation
        )
from src.error import RAGError
from src.import_utils import load_model
from src.evaluate import run_evaluation
from src.transformer import AnswerGenerator

class RAG():

    def __init__(self, raw_dir: str = 'data/raw', processed_dir: str = 'data/processed', model_name: str = 'Qwen/Qwen3-0.6B') -> None:
        self.raw_path = Path(raw_dir)
        self.processed_path = Path(processed_dir)
        self.indexer: None | Indexer = None
        self.chunked_corpus: list[Chunk] = []
        self.supported: list[str] = ['.md', '.py', '.txt']
        self.model = AnswerGenerator(model_name)
   
    def _ensure_index(self) -> Indexer:
        if self.indexer:
            return self.indexer
        chunks_file = self.processed_path / 'chunks.jsonl'
        if not chunks_file.is_file():
            raise RAGError("Index needs to run before calling any other RAG methods.\n"
                           "First run: `uv run python3 -m src index --max_chunk_size <int>`")
        try:
            with open(chunks_file, encoding='utf-8') as f:
                chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        except ValidationError as e:
            raise RAGError("Corrupted chunks.jsonl file, rebuild index first by running:\n"
                           "`uv run python3 -m src index --max_chunk_size <int?`")
        self.indexer = Indexer(corpus=chunks, loaded=True, output_path=self.processed_path)
        self.chunked_corpus = chunks
        return self.indexer
    
    def _chunk(self, max_chunk_size: int, path: Path) -> None:
        for p in sorted(path.rglob('*')):
            if p.is_file() and p.suffix in self.supported:
                self._ingest_and_chunk(max_chunk_size, p)

    def _ingest_and_chunk(self, max_chunk_size: int, path: Path) -> None:
        chunker = Chunker(path=path, max_chunk_size=max_chunk_size)
        chunker.run()
        self.chunked_corpus.extend(chunker.chunks)
    
    def index(self, max_chunk_size: int) -> None:
        files = [p for p in self.raw_path.rglob("*") if p.is_file()]
        # print(len(files), files[:5])
        self._chunk(max_chunk_size, self.raw_path)
        if self.chunked_corpus is not None:
            self.indexer = Indexer(self.chunked_corpus, False, self.processed_path)
            self.indexer.index_corpus()
            print(f"Ingestion complete! Indices saved under {self.processed_path}: \n"
                  f"{len(self.chunked_corpus)} amount of chunks with"
                  f" a chunk size of {max_chunk_size} processed and ingested.")
        else:
            raise RAGError("No corpus found to start indexing, error during chunking.")

    def _base_search(self, query: str, k: int) -> list[MinimalSource]:
        idx_results = self._ensure_index().get_ktop_idx(query, k)
        retrieved_sources = [self.chunked_corpus[i].source for i in idx_results]
        return retrieved_sources

    def _base_search_chunk(self, query: str, k: int) -> list[MinimalSource]:
        idx_results = self._ensure_index().get_ktop_idx(query, k)
        retrieved_sources = [self.chunked_corpus[i] for i in idx_results]
        return retrieved_sources
   
    def _search_with_chunks(self, query: str, k: int) -> tuple[list[Chunk], MinimalSearchResults]:
        chunks = self_base_search_chunk(query, k)
        result = MinimalSearchResults(
                question_id=str(uuid.uuid4()),
                question=query,
                retrieved_sources=[chunks.source for c in chunks])
        return (chunks, result)

    def search(self, query: str, k: int) -> MinimalSearchResults:
        result = MinimalSearchResults(
                question_id=str(uuid.uuid4()),
                question=query,
                retrieved_sources=self._base_search(query, k),
                )
        return result

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> StudentSearchResults:
        rag_dataset = load_model(dataset_path, RagDataset)
        search_results = []
        for question in rag_dataset.rag_questions:
            search_results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=self._base_search(question.question, k))
                    )
        student_search_results = StudentSearchResults(search_results=search_results,k=k)
        try:
            output_dir = Path(save_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / Path(dataset_path).name
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(student_search_results.model_dump_json(indent=2) + '\n')
        except OSError as e:
            raise RAGError(f"Could not write search results to {save_directory}: {e}") from e
        return student_search_results

    def answer(self, query: str, k: int) -> MinimalAnswer:
        chunks, search_results = self._search_with_chunks(query, k)
        return self.model.run(query, search_results, chunks)

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> StudentSearchResultsAndAnswer:
        list_of_answers = []
        student_search_results = load_model(student_search_results_path, StudentSearchResults)
        search_results = student_search_results.search_results
        for question in search_results:
            query = search_results.question
            list_of_answers.append(self.answer(query, k))
            # this is not working fully yet, reread the code, i need to pass on both the chunks
            # and the list[minimalsource] etc to the answergenerator 
        search_results_answer = StudentSearchResultsAndAnswer(
                search_results=list_of_answers, k=student_search_results.k)
        try:
            output_dir = Path(save_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / Path(student_search_results_path).name
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(search_results_answer.model_dump_json(indent=2) + '\n')
        except OSError as e:
            raise RAGError(f"Could not write search and answer results to {save_directory}: {e}") from e
        return search_results_answer

    def evaluate(student_search_results_path: str, dataset_path: str) -> Evaluation:
        return run_evaluation(student_search_results_path, dataset_path)
