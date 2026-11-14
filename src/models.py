import uuid
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import NamedTuple, Self

# represents a single source of information
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

# represents an unanswered question
class UnansweredQuestion(BaseModel):
    model_config = ConfigDict(strict=True)
    question_id: str = Field(default_factory=lambda:str(uuid.uuid4()))
    question: str

# represents an answered question
class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str

# represents a dataset of rag questions
class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

# represents a dataset of rag questions
class AnsweredRagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion]

# represents the search results
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]

# represents the answer of the search results
class MinimalAnswer(MinimalSearchResults):
    answer: str

# represents the search results and answers
class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int

class Chunk(MinimalSource):
    text: str

    @model_validator(mode='after')
    def verify(self) -> Self:
        start = self.first_character_index
        end = self.last_character_index
        if start >= end:
            raise ValueError("Chunk start index exceeds end index.")
        if len(self.text) != (end - start):
            raise ValueError("Chunk size exceeds start and end range.")
        return self

class Evaluation(BaseModel):
    eval_id: str = Field(default_factory=lambda:str(uuid.uuid4())) 
    metrics: list[tuple[int, float]]
    k: int
    count: int

class SourceKey(NamedTuple):
    file_path: str
    start: int
    end: int

