from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from pathlib import Path

from src.rag import RAGMachine
from src.models import Mode, MinimalSource
from src.log_setup import setup_logging


class CLI:
    """Subcommands for indexing, searching, answering and evaluating."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        log_file: str = "data/log",
    ) -> None:
        """Create the RAG engine, configure logging and open a console.

        Fire passes these as global flags, e.g. --raw_dir other/.

        Args:
            raw_dir: Directory holding the source files to index.
            processed_dir: Directory where the index files are written.
            log_file: Path of the debug log file.
        """
        self.rag = RAGMachine(raw_dir, processed_dir)
        setup_logging(log_file)
        self.console = Console()
        self.console.print("[bold italic purple]RAG INITIALIZATION...[/]")

    def _print_sources_table(self, sources: list[MinimalSource]) -> None:
        """Print retrieved sources as a ranked table.

        Args:
            sources: Sources in rank order, best first.
        """
        table = Table(
            title="Retrieved sources",
            box=box.SIMPLE_HEAD,
            title_justify="center",
        )
        table.add_column("#", style="yellow", justify="right")
        table.add_column(
            "File",
            style="cyan",
            overflow="ellipsis",
            no_wrap=True,
            max_width=50,
        )
        table.add_column("Range", justify="right")

        for rank, src in enumerate(sources, 1):
            table.add_row(
                str(rank),
                src.file_path,
                f"{src.first_character_index:,} - {src.last_character_index}",
            )

        self.console.print(table)

    def _print_eval_table(self, metrics: list[tuple[int, float]]) -> None:
        """Print recall@k scores as a table.

        Args:
            metrics: (k, recall) pairs, recall as a fraction in [0, 1].
        """
        table = Table(
            title="Evaluation metrics",
            box=box.SIMPLE_HEAD,
            title_justify="center",
        )
        table.add_column("RECALL@K", style="yellow", justify="right")
        table.add_column(
            "%", style="cyan", overflow="ellipsis", no_wrap=True, max_width=50
        )

        for k, score in metrics:
            table.add_row(str(k), f"{score:.3f}")

        self.console.print(table)

    def index(self, max_chunk_size: int = 2000, mode: Mode = "bm25") -> None:
        """Chunk the raw directory and build the search index.

        Args:
            max_chunk_size: Upper bound on a chunk's length, in characters.
            mode: bm25 builds only the lexical index; semantic and
                hybrid also compute and save embeddings.
        """
        self.console.rule(f"RAG · INDEX · {mode.upper()} · {max_chunk_size}")
        chunks = self.rag.index(max_chunk_size, mode)
        self.console.print(
            f"[green]Ingestion complete! Indexed [bold italic]{chunks}[/]"
            f" chunks under {self.rag.processed_path}[/]."
        )

    def search(self, query: str, k: int, mode: Mode = "bm25") -> None:
        """Retrieve the top k chunks for one question and print them.

        Args:
            query: The question. Converted to str because Fire parses
                bare numbers as ints.
            k: Number of chunks to retrieve.
            mode: Retrieval method: bm25, semantic or hybrid.
        """
        query = str(query)
        self.console.rule(f"RAG · SEARCH · {mode.upper()}")
        self.console.print(
            Panel(
                query,
                title="Question",
                border_style="bold",
                expand=False,
            )
        )
        search_results = self.rag.search(query, k, mode)
        self._print_sources_table(search_results.retrieved_sources)

    def answer(self, query: str, k: int, mode: Mode = "bm25") -> None:
        """Retrieve context for one question and print a generated answer.

        Args:
            query: The question. Converted to str because Fire parses
                bare numbers as ints.
            k: Number of chunks to retrieve as context.
            mode: Retrieval method: bm25, semantic or hybrid.
        """
        query = str(query)
        self.console.rule(f"RAG · ANSWER · {mode.upper()}")
        self.console.print(
            Panel(
                query,
                title="Question",
                border_style="bold",
                expand=False,
            )
        )
        answer = self.rag.answer(query, k, mode)
        self._print_sources_table(answer.retrieved_sources)
        self.console.print(
            Panel(
                answer.answer,
                title=f"Answer (k = {len(answer.retrieved_sources)})",
                border_style="bold",
                expand=False,
                padding=(1, 1),
            )
        )

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        mode: Mode = "bm25",
    ) -> None:
        """Run retrieval for every question in a dataset and save results.

        The output file has the same name as the dataset and is written
        to save_directory.

        Args:
            dataset_path: JSON file matching RagDataset.
            k: Number of chunks to retrieve per question.
            save_directory: Directory to write the results into.
            mode: Retrieval method: bm25, semantic or hybrid.
        """
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
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for saved search results and save them.

        Uses the sources already chosen by search_dataset rather than
        searching again. The output file keeps the input file's name.

        Args:
            student_search_results_path: JSON file written by
                search_dataset.
            save_directory: Directory to write the answers into.
        """
        self.console.rule("RAG · ANSWER_DATASET")
        _ = self.rag.answer_dataset(
            student_search_results_path,
            save_directory,
        )
        out_file = (
            Path(save_directory) / Path(student_search_results_path).name
        )
        self.console.print(
            "[bold purple]Saved student search results"
            f" and answers to: {out_file}.[/]"
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Score saved search results against a reference dataset.

        Prints recall@k for each k in 1, 3, 5 and 10 that does not exceed
        the k the results were produced with.

        Args:
            student_search_results_path: JSON file written by
                search_dataset.
            dataset_path: JSON file matching AnsweredRagDataset,
                with the reference sources for each question.
        """
        self.console.rule("RAG · EVALUATE")
        evaluation_scores = self.rag.evaluate(
            student_search_results_path, dataset_path
        )
        metrics = evaluation_scores.metrics
        self._print_eval_table(metrics)
