from tqdm import tqdm
from pathlib import Path
from pydantic import ValidationError
import uuid
import logging
import time

from src.chunker import Chunker
from src.index import Indexer
from src.models import (
    MinimalSource,
    Chunk,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    RagDataset,
    Evaluation,
    SourceKey,
    Mode,
)
from src.error import RAGError, EmptyQueryError
from src.import_utils import load_model
from src.evaluate import run_evaluation
from src.transformer import AnswerGenerator
from src.tokenizer import tokenize

log = logging.getLogger(__name__)


class RAGMachine:

    def __init__(self, raw_dir: str, processed_dir: str) -> None:
        self.raw_path = Path(raw_dir)
        self.processed_path = Path(processed_dir)
        self.indexer: None | Indexer = None
        self.chunked_corpus: list[Chunk] = []
        self.chunk_by_source: dict[SourceKey, Chunk] = {}
        self.supported: list[str] = [".md", ".py", ".txt"]
        self.model: AnswerGenerator | None = None
        self.MAX_CHUNK_SIZE: int = 2000

    def _check_k(self, k: int) -> None:
        if not isinstance(k, int) or isinstance(k, bool):
            raise RAGError("Parameter k has to be a positive integer.")
        if k <= 0:
            raise RAGError("Parameter k has to be a positive integer.")
        if k >= len(self.chunked_corpus):
            raise RAGError(
                "Parameter k cannot be larger than"
                " the amount of chunks in the source corpus."
            )

    def _ensure_chunks(self) -> list[Chunk]:
        if self.chunked_corpus:
            return self.chunked_corpus
        chunks_file = self.processed_path / "chunks.jsonl"
        if not chunks_file.is_file():
            raise RAGError(
                "Index needs to run before calling any other RAG methods.\n"
                "First run: "
                "`uv run python3 -m src index --max_chunk_size <int>`"
            )
        try:
            with open(chunks_file, encoding="utf-8") as f:
                self.chunked_corpus = [
                    Chunk.model_validate_json(line)
                    for line in f
                    if line.strip()
                ]
        except ValidationError as e:
            raise RAGError("Corrupted chunks.jsonl, rebuild the index.") from e
        self.chunk_by_source = {
            SourceKey(
                c.file_path, c.first_character_index, c.last_character_index
            ): c
            for c in self.chunked_corpus
        }
        return self.chunked_corpus

    def _ensure_index(self, mode: Mode) -> Indexer:
        if self.indexer:
            return self.indexer
        chunks = self._ensure_chunks()
        self.indexer = Indexer(
            corpus=chunks,
            loaded=True,
            output_path=self.processed_path,
            mode=mode,
        )
        return self.indexer

    def _chunk(self, max_chunk_size: int, path: Path) -> None:
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix in self.supported
        )
        for p in tqdm(files, desc="Chunking corpus..."):
            self._ingest_and_chunk(max_chunk_size, p)

    def _ingest_and_chunk(self, max_chunk_size: int, path: Path) -> None:
        chunker = Chunker(path=path, max_chunk_size=max_chunk_size)
        chunker.run()
        self.chunked_corpus.extend(chunker.chunks)

    def index(self, max_chunk_size: int, mode: Mode = "bm25") -> int:
        start = time.perf_counter()
        if (
            not isinstance(max_chunk_size, int)
            or isinstance(max_chunk_size, bool)
            or not 0 < max_chunk_size <= self.MAX_CHUNK_SIZE
        ):
            raise RAGError(
                "Max chunk size has to be an integer between"
                f" 1 and {self.MAX_CHUNK_SIZE}."
            )
        self._chunk(max_chunk_size, self.raw_path)
        elapsed = time.perf_counter() - start
        log.info(f"Chunking time: {elapsed:.3f}s.")
        if self.chunked_corpus:
            self.processed_path.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            self.indexer = Indexer(
                self.chunked_corpus, False, self.processed_path, mode
            )
            self.indexer.index_corpus(mode)
            elapsed = time.perf_counter() - start
            log.info(f"Indexing time: {elapsed:.3f}s.")
        else:
            raise RAGError(
                "No corpus found to start indexing, error during chunking."
            )
        return len(self.chunked_corpus)

    @staticmethod
    def _get_minimal_source(chunk: Chunk) -> MinimalSource:
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )

    def _base_search(self, query: str, k: int, mode: Mode) -> list[int]:
        if mode != 'semantic' and not tokenize(query):
            raise EmptyQueryError(f"Query '{query}' has no searchable terms"
                                  " after stopword removal.")
        return self._ensure_index(mode).get_ktop_idx(query, k, mode)

    def _search_with_chunks(
        self, query: str, k: int, mode: Mode
    ) -> tuple[list[Chunk], MinimalSearchResults]:
        sources_idx = self._base_search(query, k, mode)
        chunks = [self.chunked_corpus[i] for i in sources_idx]
        result = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=[self._get_minimal_source(c) for c in chunks],
        )
        return (chunks, result)

    def search(
        self, query: str, k: int, mode: Mode = "bm25"
    ) -> MinimalSearchResults:
        self._ensure_chunks()
        self._check_k(k)
        sources_idx = self._base_search(query, k, mode)
        sources = [
            self._get_minimal_source(self.chunked_corpus[i])
            for i in sources_idx
        ]
        result = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=sources,
        )
        for rank, src in enumerate(sources, 1):
            log.debug(
                "#%d %s [%d:%d]",
                rank,
                src.file_path,
                src.first_character_index,
                src.last_character_index,
            )
        return result

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        mode: Mode = "bm25",
    ) -> StudentSearchResults:
        self._ensure_chunks()
        self._check_k(k)
        rag_dataset = load_model(dataset_path, RagDataset)
        search_results = []
        start = time.perf_counter()
        for q in tqdm(range(len(rag_dataset.rag_questions))):
            question = rag_dataset.rag_questions[q]
            try:
                sources_idx = self._base_search(question.question, k, mode)
            except EmptyQueryError:
                log.warning("No searchable terms in "
                            f"question {question.question_id}"
                            ", saving it with no retrieved sources.")
                sources_idx = []
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[
                        self._get_minimal_source(self.chunked_corpus[i])
                        for i in sources_idx
                    ],
                )
            )
        elapsed = time.perf_counter() - start
        log.info(
            f"Retrieval throughput {elapsed}s for"
            f" {len(rag_dataset.rag_questions)} questions."
        )
        student_search_results = StudentSearchResults(
            search_results=search_results, k=k
        )
        try:
            output_dir = Path(save_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / Path(dataset_path).name
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(
                    student_search_results.model_dump_json(indent=2) + "\n"
                )
        except OSError as e:
            raise RAGError(
                f"Could not write search results to {save_directory}: {e}"
            ) from e
        return student_search_results

    def _generate_answer(
        self, search_results: MinimalSearchResults, chunks: list[Chunk]
    ) -> MinimalAnswer:
        if self.model is None:
            self.model = AnswerGenerator()
        return self.model.run(search_results, chunks)

    def answer(self, query: str, k: int, mode: Mode = "bm25") -> MinimalAnswer:
        self._ensure_chunks()
        self._check_k(k)
        chunks, search_results = self._search_with_chunks(query, k, mode)
        return self._generate_answer(search_results, chunks)

    def _get_chunks_from_sources(
        self, sources: list[MinimalSource]
    ) -> list[Chunk]:
        def _key(src: MinimalSource) -> SourceKey:
            return SourceKey(
                src.file_path,
                src.first_character_index,
                src.last_character_index,
            )

        self._ensure_chunks()
        try:
            return [self.chunk_by_source[_key(s)] for s in sources]
        except KeyError as e:
            raise RAGError(
                "Search results do not match current index,"
                f" re-run 'search_dataset': {e}"
            ) from e

    def _answer_from_search_results(
        self, query: str, search_results: MinimalSearchResults
    ) -> MinimalAnswer:
        chunks = self._get_chunks_from_sources(
            search_results.retrieved_sources
        )
        return self._generate_answer(search_results, chunks)

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> StudentSearchResultsAndAnswer:
        list_of_answers = []
        student_search_results = load_model(
            student_search_results_path, StudentSearchResults
        )
        search_results = student_search_results.search_results
        for i in tqdm(
            range(len(search_results)), desc="Generating answers..."
        ):
            question = search_results[i]
            query = question.question
            list_of_answers.append(
                self._answer_from_search_results(query, question)
            )
        search_results_answer = StudentSearchResultsAndAnswer(
            search_results=list_of_answers, k=student_search_results.k
        )
        try:
            output_dir = Path(save_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / Path(student_search_results_path).name
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(search_results_answer.model_dump_json(indent=2) + "\n")
        except OSError as e:
            raise RAGError(
                "Could not write search and answer results"
                f" to {save_directory}: {e}"
            ) from e
        return search_results_answer

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> Evaluation:
        return run_evaluation(student_search_results_path, dataset_path)
