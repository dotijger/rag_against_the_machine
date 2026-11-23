import sys
import fire
import logging

from src.cli import CLI
from src.error import RAGError

log = logging.getLogger("src.cli")


def main() -> int:
    """Run the CLI and map the outcome to a process exit code.

    Returns:
        0 on success, 1 on a :class:`RAGError` or any unexpected
        exception, and 130 when the user interrupts with Ctrl+C.
    """
    try:
        fire.Fire(CLI)
    except RAGError as e:
        log.error(e)
        return 1
    except KeyboardInterrupt:
        log.warning("User interrupted RAG session.")
        return 130
    except Exception as e:
        log.critical(f"Unexpected error occurred: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
