from pydantic import ValidationError, BaseModel
from pathlib import Path
from typing import Any
import json

from src.error import ImporterError, RAGError
from src.models import MinimalSource, UnansweredQuestion, AnsweredQuestion, RagDataset

def _is_readable_path(path: str) -> bool:
    if not isinstance(path, (str, Path)):
        return False

    to_import = Path(path)
    
    if not to_import.is_file():
        return False

    try:
        with to_import.open('r'):
            return True
    except (PermissionError, OSError):
        return False

def json_load(path: str) -> Any:
    if not _is_readable_path(path):
        raise ImporterError(f"{path} is not a readable path, aborting.")
    with open(path, encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ImporterError(e)
        except FileNotFoundError as e:
            raise ImporterError(e)
        except IsADirectoryError as e:
            raise ImporterError(e)
    if not isinstance(data, dict):
        raise ImporterError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data

def load_model[M: BaseModel](path: str, model_cls: type[M]) -> M:
    try:
        _ = json_load(path)
    except ImporterError as e:
        raise ImporterError(e)
    try:
        return model_cls.model_validate_json(Path(path).read_text(encoding='utf-8'))
    except OSError as e:
        raise RAGError(f"Cannot read {path}: {e}") from e
    except ValidationError as e:
        raise RAGError(f"{path} does not match {model_cls.__name__}:\n{e}") from e

## the below functions are not in use anymore but still here for historical purposes 
def dict_to_minimal_source(source_data: dict[str, Any]) -> MinimalSource:
    return MinimalSource(
            file_path=source_data['file_path'],
            first_character_index=source_data['first_character_index'],
            last_character_index=source_data['last_character_index']
            )

def get_unanswered_questions(data: list[dict[str, Any]]) -> list[UnansweredQuestion]:
    return [UnansweredQuestion(question_id=q['question_id'],question=q['question']) for q in data]

def get_answered_questions(data: list[dict[str, Any]]) -> list[UnansweredQuestion]:
    return [
            AnsweredQuestion(
                question_id=q['question_id'],
                question=q['question'],
                sources=[dict_to_minimal_source(source) for source in q['sources']],
                answer=q['answer']
                ) 
            for q in data
            ]

def import_rag_dataset(path: str) -> RagDataset:
    dataset = json_load(path)
    try:
        rag_dataset = RagDataset.model_validate_json(Path(path).read_text(encoding='utf-8'))
    except ValidationError as e:
        raise ImporterError(f"{path} does not have a valid RAG dataset structure.")
    questions = dataset.get('rag_questions')
    if len(questions) == 0:
        raise ImporterError(f'{path} does not contain any questions.')
    if questions[0].get('answer'):
        return RagDataset(
                rag_questions=get_answered_questions(questions)
                )
    return RagDataset(
            rag_question=get_unanswered_questions(questions)
            )

def import_rag_questions(path: str) -> list[UnansweredQuestion]:
    dataset = json_load(path)
    if dataset.get('rag_questions') is None:
        raise ImportError("Dataset does not contain RAG questions, aborting.")
    questions = dataset.get('rag_questions')
    rag_questions = []
    for question in questions:
        try:
            rag_questions.append(
                    UnansweredQuestion(
                        question_id=question.get('question_id'),
                        question=question.get('question'),
                        )
                    )
        except ValidationError as e:
            raise ImporterError(e)
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
            raise ImporterError("Question id is not a string, aborting.")
        if q_id is None:
            raise ImporterError("Dataset entry does not contain question id, aborting.")
        if true_sources.get(q_id):
            raise ImporterError("Dataset entry is a duplicate, aborting.")
        true_sources[q_id] = []
        q_sources = question.get('sources')
        if q_sources is None:
            raise ImporterError("Dataset entry does not contain listed sources, aborting.")
        for source in q_sources:
            try:
                true_sources[q_id].append(
                        MinimalSource(
                            file_path=source['file_path'],
                            first_character_index=source['first_character_index'],
                            last_character_index=source['last_character_index'],
                            )
                        )
            except (KeyError, ValidationError) as e:
                raise ImporterError(f"Could not convert source to MinimalSource: {e}, aborting.")
    return true_sources 
