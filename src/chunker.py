# chunking strategies for 1 python code chunking (.py) and 2 markdown / text chunking (.txt, .md)
# use of TD-IDF -> BM25 lexical retieving methods

from pydantic import BaseModel, model_validator, ValidationError
from typing import Self
from pathlib import Path
import re

from src.utils import read_source

class ChunkError(BaseException):
    pass

class Chunk(BaseModel):
    file_path: Path
    start: int
    end: int
    text: str

    @model_validator(mode='after')
    def verify(self) -> Self:
        if self.start >= self.end:
            raise ValueError("Chunk start index exceeds end index.")
        if len(self.text) != (self.end - self.start):
            raise ValueError("Chunk size exceeds start and end range.")

class Chunker():
    SUPPORTED = ['.txt', '.md', '.py']

    def __init__(self, path: Path, max_chunk_size: int = 1500) -> None:
        self.path = path
        self.max_chunk_size = max_chunk_size
        self.chunks: list[Chunk] = []
    
    def _read_source(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            self.body = f.read()

    # added start and stop that default to the full unchunked file body but
    # useful for if i need to use this for a select part of the text -- so parts
    # that do not fall under the ast tree or are too big etc - final resort
    def _window_chunker(self, start: int = 0, stop: int = len(self.body)) -> None:
        step = self.max_chunk_size
        for idx in range(start, stop, step):
            try:
                self.chunks.append(
                        Chunk(file_path=self.path,
                              start=idx,
                              end=min(idx + step, stop),
                              text=self.body[idx:idx+step])
                        )
            except ValidationError as e:
                raise ValueError(e)
    
    def run(self) -> None:
        if self.path.suffix not in self.SUPPORTED:
            raise ChunkError("File type not supported by chunking methods, aborting.")
        self._read_source()
        self._chunkify()


    def _chunkify(self, suffix: str) -> None:
        if self.path.suffix == '.txt':
            pass
        elif self.path.suffix == '.md':
            pass
        elif self.path.suffix == '.py':
            # if ast.parse raises SyntaxError, default to _window_chunker() for entire file
            pass

    def _md(self) -> None:
        # need to make a separate class probably 
        heading = re.compile(r"*#{1, 6}\s")
        fence = re.compile(r"*(```|~~~)")
        lines = self.body.split('\n')
        cuts = [0]
        offset = 0
        in_fence = False
        for line in lines:
            if fench.match(line):
                in_fence = not in_fence
            elif not in_fence and heading.match(line) and offset > 0:
                cuts.append(offset)
            offset += len(line)
        cuts.append(len(source))
        # return cuts


if __name__ == "__main__":
    p = Path('test')
    for p in list(p.glob('*.py')):
        chunks = chunkify(p)
        for c in chunks:
            print(f"{c.file_path=}\n{c.start=}\n{c.end=}\n{c.text=}")

