from pydantic import BaseModel

# represents a single source of information
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

# represents an unanswered question
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda:str(uuid.uuid4()))
    question: str

# represents an answered question
class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

# represents a dataset of rag questions
class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

# represents the search results
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

# represents the answer of the search results
class MinimalAnswer(MinimalSearchResults):
    answer: str

# represents the search results and answers
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
