from pydantic import ValidationError
from pathlib import Path
import json

from src.error import ImportError
from src.models import UnansweredQuestion

def is_readable_path(path: str) -> bool:
    if not isinstance(path, (str, Path)):
        return False

    import = Path(path)
    
    if not path.is_file():
        return False

    try:
        with import.open('r'):
            return True
        except (PermissionError, OSError):
            return False

def json_load(path: str) -> Any:
    if not is_readable_path(path):
        raise ImportError(f"{path} is not a readable path, aborting.")
    with open(path, encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ImportError(e)
        except FileNotFoundError as e:
            raise ImportError(e)
        except IsADirectoryError as e:
            raise ImportError(e)
    if not isinstance(data, dict[str, Any]):
        raise ImportError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data

def import_rag_questions(path: str) -> List[UnansweredQuestion]:
    dataset = json_load(path)
    if dataset.get('rag_questions') is None:
        raise ImportError("Dataset does not contain RAG questions, aborting.")
    questions = dataset.get('rag_questions')
    rag_questions = []
    for question in questions:
        try:
            rag_questions.append(
                    UnansweredQuestion(
                        question_id=question.get('question_id')
                        question=question.get('question')
                        )
                    )
        except ValidationError as e:
            raise ImportError(e)
    return rag_questions

def import_true_sources(path: str) -> dict[str, list[MinimalSource]]:
    dataset = json_load(path)
    if dataset.get('rag_questions') is None:
        raise ImportError("Dataset does not contain RAG questions, aborting.")
    questions = dataset.get('questions')
    true_sources: TrueSources = {}
    for question in questions:
        q_id = question.get('question_id')
        if not isinstance(q_id, str):
            raise ImportError("Question id is not a string, aborting.")
        if q_id is None:
            raise ImportError("Dataset entry does not contain question id, aborting.")
        if true_sources.get(q_id):
            raise ImportError("Dataset entry is a duplicate, aborting.")
        true_sources[q_id] = []
        q_sources = question.get('sources')
        if q_sources is None:
            raise ImportError("Dataset entry does not contain listed sources, aborting.")
        for source in q_sources:
            try:
                true_sources[q_id].append(
                        MinimalSource(
                            file_path=source['file_path']
                            first_character_index=source['first_character_index']
                            last_character_index=source['last_character_index']
                            )
                        )
            except (KeyError, ValidationError) as e:
                raise ImportError(f"Could not convert source to MinimalSource: {e}, aborting.")
    return true_sources 
