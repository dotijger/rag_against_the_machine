from rich.console import Console
from rich.theme import Theme
from rich.logging import RichHandler
import logging
from pathlib import Path


def setup_logging() -> None:
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

    log_file = Path("data/log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(log_file, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(file)
