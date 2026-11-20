from pydantic import ValidationError, BaseModel
from pathlib import Path
from typing import Any
import json

from src.error import ImporterError


def _is_readable_path(path: str) -> bool:
    if not isinstance(path, (str, Path)):
        return False

    to_import = Path(path)

    if not to_import.is_file():
        return False

    try:
        with to_import.open("r"):
            return True
    except (PermissionError, OSError):
        return False


def json_load(path: str) -> Any:
    if not _is_readable_path(path):
        raise ImporterError(f"{path} is not a readable path, aborting.")
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ImporterError(e)
        except FileNotFoundError as e:
            raise ImporterError(e)
        except IsADirectoryError as e:
            raise ImporterError(e)
    if not isinstance(data, dict):
        raise ImporterError(
            f"{path}: expected a JSON object, got {type(data).__name__}"
        )
    return data


def load_model[M: BaseModel](path: str, model_cls: type[M]) -> M:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ImporterError(f"Cannot read {path}: {e}") from e
    try:
        return model_cls.model_validate_json(raw)
    except ValidationError as e:
        raise ImporterError(
            f"{path} does not match {model_cls.__name__}:\n{e}"
        ) from e
