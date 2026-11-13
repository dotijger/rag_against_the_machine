import uuid
from pydantic import BaseModel, Field, ConfigDict
from typing import TypedDict

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

class Evaluation(BaseModel):
    eval_id: str = Field(default_factory=lambda:str(uuid.uuid4())) 
    metrics: list[tuple[int, float]]
    k: int
    count: int

