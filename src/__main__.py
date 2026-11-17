from rich.console import Console
import sys
import fire


from src.cli import CLI
from src.error import (
        RAGError,
        IndexError,
        ChunkError,
        ImporterError,
        )

if __name__ == "__main__":
    c = Console()
    try:
        fire.Fire(CLI())
    except (ValueError, RAGError, IndexError, ChunkError, ImporterError, TypeError) as e:
        c.print(e, style="bold red")
        sys.exit(1)
    except KeyboardInterrupt as e:
        c.print(e, style="bold red")
        sys.exit(1)
