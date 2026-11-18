from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from pathlib import Path

from src.rag import RAGMachine
from src.error import RAGError
from src.models import Mode, MinimalSource
from src.log_setup import setup_logging


class CLI():

    def __init__(self,
                 raw_dir: str = 'data/raw',
                 processed_dir: str = 'data/processed',
                 log_level: str = 'warning',
                 file_path: str | None = None,
                 ) -> None:
        self.rag = RAGMachine(raw_dir, processed_dir)
        setup_logging()
        self.console = Console()
        self.console.print("[bold italic purple]RAG INITIALIZATION...[/]")

    def _print_sources_table(self, sources: list[MinimalSource]) -> None:
        table = Table(
                title='Retrieved sources',
                box=box.SIMPLE_HEAD,
                title_justify='center')
        table.add_column('#', style='yellow', justify='right')
        table.add_column(
                'File', style='cyan', overflow='ellipsis',
                no_wrap=True, max_width=50)
        table.add_column('Range', justify='right')

        for rank, src in enumerate(sources, 1):
            table.add_row(
                    str(rank),
                    src.file_path,
                    f"{src.first_character_index:,} - "
                    f"{src.last_character_index}",
                    )

        self.console.print(table)

    def _print_eval_table(self, metrics: list[tuple[int, float]]) -> None:
        table = Table(
                title='Evaluation metrics',
                box=box.SIMPLE_HEAD,
                title_justify='center')
        table.add_column('RECALL@K', style='yellow', justify='right')
        table.add_column('%', style='cyan', overflow='ellipsis',
                         no_wrap=True, max_width=50)

        for k, score in metrics:
            table.add_row(
                    str(k),
                    f"{score:.3f}")

        self.console.print(table)

    def _check_k(self, k: int) -> None:
        if k <= 0:
            raise RAGError("Parameter k has to be a positive integer.")

    def index(self, max_chunk_size: int, mode: Mode = 'bm25') -> None:
        if max_chunk_size <= 0:
            raise RAGError("Max chunk size has to be a positive integer.")
        self.console.rule(f"RAG · INDEX · {mode.upper()} · {max_chunk_size}")
        chunks = self.rag.index(max_chunk_size, mode)
        self.console.print(
                f"[green]Ingestion complete! Indexed [bold italic]{chunks}[/]"
                f" chunks under {self.rag.processed_path}[/].")

    def search(self, query: str, k: int, mode: Mode = 'bm25') -> None:
        self._check_k(k)
        self.console.rule(f"RAG · SEARCH · {mode.upper()}")
        self.console.print(Panel(
            query,
            title='Question',
            border_style="bold",
            expand=False,
            ))
        search_results = self.rag.search(query, k, mode)
        self._print_sources_table(search_results.retrieved_sources)

    def answer(self, query: str, k: int, mode: Mode = 'bm25') -> None:
        self._check_k(k)
        self.console.rule(f"RAG · ANSWER · {mode.upper()}")
        self.console.print(Panel(
            query,
            title='Question',
            border_style="bold",
            expand=False,
            ))
        answer = self.rag.answer(query, k, mode)
        self._print_sources_table(answer.retrieved_sources)
        self.console.print(Panel(
            answer.answer,
            title=f'Answer (k = {len(answer.retrieved_sources)})',
            border_style="bold",
            expand=False,
            padding=(1, 1),
            ))

    def search_dataset(
            self,
            dataset_path: str,
            k: int,
            save_directory: str,
            mode: Mode = "bm25",
            ) -> None:
        self._check_k(k)
        self.console.rule(f"RAG · SEARCH_DATASET · {mode.upper()}")
        _ = self.rag.search_dataset(
                dataset_path,
                k,
                save_directory,
                mode,
                )
        out_file = Path(save_directory) / Path(dataset_path).name
        self.console.print(
                f"[bold purple]Saved student search results to: {out_file}.[/]"
                )

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
            ) -> None:
        self.console.rule("RAG · ANSWER_DATASET")
        _ = self.rag.answer_dataset(
                student_search_results_path,
                save_directory,
                )
        out_file = (Path(save_directory) /
                    Path(student_search_results_path).name)
        self.console.print(
                "[bold purple]Saved student search results"
                f" and answers to: {out_file}.[/]"
                )

    def evaluate(self,
                 student_search_results_path: str,
                 dataset_path: str) -> None:
        self.console.rule("RAG · EVALUATE")
        evaluation_scores = self.rag.evaluate(
                student_search_results_path, dataset_path)
        metrics = evaluation_scores.metrics
        self._print_eval_table(metrics)
