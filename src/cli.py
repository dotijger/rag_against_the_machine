from pathlib import Path

from src.rag import RAG
from src.error import RAGError
from src.models import Mode
from src.logging import RAGLogger

class CLI():

    def __init__(self, raw_dir: str = 'data/raw', processed_dir: str = 'data/processed') -> None:
        self.log = logger.getLogger(__name__)
        self.rag = Rag(raw_dir, processed_dir)

    def index(self, max_chunk_size: int, mode: Mode = 'bm25') -> None:
        self.rag.index(max_chunk_size, mode)

    def search(self, query: str, k: int, mode: Mode = 'bm25') -> None:
        search_results = self.rag.search(query, k, mode)

    def answer(self, query: str, k: int, mode: Mode = 'bm25') -> None:
        answer = self.rag.answer(query, k, mode)

    def search_dataset(
            self,
            dataset_path: str,
            k: int,
            save_directory: str,
            mode: Mode = "bm25"
            ) -> None:
        student_search_results = self.rag.search_dataset(
                dataset_path,
                k,
                save_directory,
                mode,
                )

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
            ) -> None:
        student_search_results_answers = self.rag.answer_dataset(
                student_search_results_path,
                save_directory,
                )

    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        evaluation_scores = self.rag.evaluate(student_search_results_path, dataset_path)
        metrics = evaluation_scores.metrics
        print("Evaluation Results\n========================================")
        for metric in metrics:
            print(f"Recall@{metric[0]}: {metric[1]:.3f} ", end="")
