import uuid
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import NamedTuple, Self, Literal


class MinimalSource(BaseModel):
    """A character range within one source file.

    Attributes:
        file_path: Path of the file the range belongs to.
        first_character_index: Offset of the first character.
        last_character_index: Offset one past the last character.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A dataset question with no reference answer.

    Validation is strict, so types are not coerced.

    Attributes:
        question_id: Unique ID; a random UUID if not given.
        question: The question text.
    """

    model_config = ConfigDict(strict=True)
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A dataset question with its reference sources and answer.

    Attributes:
        sources: The ranges a good retriever should find.
        answer: The reference answer.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, which may or may not have answers.

    Attributes:
        rag_questions: The questions.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class AnsweredRagDataset(BaseModel):
    """A dataset where every question has reference sources and an answer.

    Attributes:
        rag_questions: The questions; used as ground truth for evaluation.
    """

    rag_questions: list[AnsweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question.

    Attributes:
        question_id: ID of the question.
        question: The question text.
        retrieved_sources: Retrieved ranges, best match first.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results for one question plus the generated answer.

    Attributes:
        answer: The answer produced by the language model.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for a whole dataset, as saved to disk.

    Attributes:
        search_results: One entry per question.
        k: Number of sources retrieved per question.
    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results and generated answers for a whole dataset.

    Attributes:
        search_results: One entry per question, each with an answer.
        k: Number of sources retrieved per question.
    """

    search_results: list[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """A source range together with its text.

    Attributes:
        text: The file contents within the range.
    """

    text: str

    @model_validator(mode="after")
    def verify(self) -> Self:
        start = self.first_character_index
        end = self.last_character_index
        if start >= end:
            raise ValueError("Chunk start index exceeds end index.")
        if len(self.text) != (end - start):
            raise ValueError("Chunk size exceeds start and end range.")
        return self


class Evaluation(BaseModel):
    """The outcome of evaluating search results against a dataset.

    Attributes:
        eval_id: Unique ID; a random UUID if not given.
        metrics: (k, recall@k) pairs.
        k: The k the search results were produced with.
        count: Number of questions evaluated.
    """

    eval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metrics: list[tuple[int, float]]
    k: int
    count: int


class SourceKey(NamedTuple):
    """Hashable key for looking up a chunk by its file and range.

    Attributes:
        file_path: Path of the file.
        start: Offset of the first character.
        end: Offset one past the last character.
    """

    file_path: str
    start: int
    end: int


Mode = Literal["bm25", "semantic", "hybrid"]
"""Retrieval method: lexical, embedding-based, or both fused with RRF."""
