from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler
import logging
from pathlib import Path


def setup_logging(file_path: str) -> None:
    """Send src logs to the terminal and to a file.

    INFO and above go to stderr through Rich, so they never mix with the
    command's normal output on stdout. DEBUG and above go to the log file
    with timestamps and logger names. Existing handlers are removed first,
    so calling this twice does not duplicate messages, and propagation to
    the root logger is turned off so other libraries' handlers stay out.

    Args:
        file_path: Path of the log file; parent directories are created
            if needed. The file is appended to.
    """
    logger = logging.getLogger("src")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()

    log_theme = Theme(
        {
            "logging.level.info": "cyan",
            "logging.level.error": "bold red",
            "log.time": "dim",
        }
    )
    console = RichHandler(
        console=Console(stderr=True, theme=log_theme),
        log_time_format="[%X]",
        show_path=False,
    )
    console.setFormatter(logging.Formatter("%(message)s"))
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    log_file = Path(file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(log_file, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(file)
