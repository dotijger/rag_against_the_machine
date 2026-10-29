# chunking strategies for 1 python code chunking (.py) and 2 markdown / text chunking (.txt, .md)
# use of TD-IDF -> BM25 lexical retieving methods

from pydantic import BaseModel, model_validator, ValidationError
from typing import Self

class Chunk(BaseModel):
    file_path: str
    start: int
    end: int
    text: str

    def verify(self) -> Self:
        if start > end:
            raise ValidationError("Chunk start index exceeds end index.")
        if len(text) > (end - start):
            raise ValidationError("Chunk size exceeds start and end range.")

    

