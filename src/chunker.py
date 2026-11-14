# chunking strategies for 1 python code chunking (.py) and 2 markdown / text chunking (.txt, .md)
# use of TD-IDF -> BM25 lexical retieving methods

from pydantic import BaseModel, model_validator, ValidationError
from typing import Self
from pathlib import Path
import re

from src.error import ChunkError
from src.models import MinimalSource, Chunk

class Chunker():
    SUPPORTED = ['.txt', '.md', '.py']

    def __init__(self, path: str, max_chunk_size: int = 1500) -> None:
        self.path = Path(path)
        self.max_chunk_size = max_chunk_size
        self.chunks: list[Chunk] = []
        self.body: str = ""
    
    def _read_source(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            self.body = f.read()

    # added start and stop that default to the full unchunked file body but
    # useful for if i need to use this for a select part of the text -- so parts
    # that do not fall under the ast tree or are too big etc - final resort
    def _window_chunker(self, start: int = 0, stop: int = -1) -> None:
        step = self.max_chunk_size
        if stop == -1:
            stop = len(self.body)
        for idx in range(start, stop, step):
            try:
                self.chunks.append(Chunk(
                            file_path=str(self.path),
                            first_character_index=idx,
                            last_character_index=min(idx + step, stop),                          
                            text=self.body[idx:idx+step]))
            except ValidationError as e:
                raise ValueError(e)
    
    def run(self) -> None:
        if self.path.suffix not in self.SUPPORTED:
            print("File type not supported, skipping file")
            return
            # raise ChunkError("File type not supported by chunking methods, skipping file.")
        self._read_source()
        if len(self.body) == 0:
            return
        if self.path.suffix == '.txt':
            self._window_chunker()
        elif self.path.suffix == '.md':
            self._md()
        elif self.path.suffix == '.py':
            # print(f"running {self.path=}")
            self._py()

    def _py(self) -> None:
        starters = ('def', 'async def', 'class', '@')
        lines = self.body.split('\n')
        cuts = []
        offset = 0
        previous = ""
        for line in lines:
            starter = line.startswith(starters)
            after_dec = previous.startswith("@") and not line.startswith("@")
            if starter and not after_dec and offset > 0:
                last = cuts[-1] if cuts else 0
                if offset - last > 100:
                    cuts.append(offset)
            if line.strip():
                previous = line
            offset += len(line) + 1
        cuts.append(len(self.body))
        self._chunkify(self._merge(cuts))


    def _md(self) -> None:
        heading = re.compile(r"#{1,6}\s")
        code_header = re.compile(r"(```|~~~)")
        lines = self.body.split('\n')
        cuts = []
        offset = 0
        code_block = False
        for line in lines:
            if code_header.match(line):
                code_block = not code_block
            elif not code_block and heading.match(line) and offset > 0:
                last = cuts[-1] if cuts else 0
                if offset - last > 100:
                    cuts.append(offset)
            offset += len(line) + 1
        cuts.append(len(self.body))
        self._chunkify(self._merge(cuts))
    
    def _windows(self, start: int, end: int) -> list[tuple[int, int]]:
        step = self.max_chunk_size
        ranges = []
        while end - start > step:
            newline = self.body.rfind("\n", start, start + step)
            cut = newline + 1 if newline != -1 else start + step
            ranges.append((start, cut))
            start = cut
        ranges.append((start, end))
        return ranges

    def _merge(self, cuts: list[int]) -> list[int]:
        merged: list[int] = []
        start = 0
        pending: int | None = None
        for cut in cuts:
            if cut - start <= self.max_chunk_size:
                pending = cut
                continue
            if pending is not None:
                merged.append(pending)
                start = pending
            pending = cut
            if cut - start > self.max_chunk_size:
                merged.append(cut)
                start = cut
                pending = None
        if pending is not None:
            merged.append(pending)
        return merged

    def _chunkify(self, cuts: list[int]) -> None:
        previous = 0
        # print(f"{cuts=}")
        for cut in cuts:
            for start, end in self._windows(previous, cut):
                try:
                    self.chunks.append(Chunk(
                                file_path=str(self.path),
                                first_character_index=start,
                                last_character_index=end,
                                text=self.body[start:end]))
                except ValidationError as e:
                    raise ValueError(e)
            previous = cut

if __name__ == "__main__":
    p = Path('test')
    for p in list(p.glob('*.md')):
        print(f"{p=}")
        ch = Chunker(p)
        ch.run()
        for c in ch.chunks:
            print(f"{c.file_path=}\n{c.start=}\n{c.end=}\n{c.text=}")

    p = Path('test')
    for p in list(p.glob('*.py')):
        print(f"{p=}")
        ch = Chunker(p)
        ch.run()
        for c in ch.chunks:
            print(f"{c.file_path=}\n{c.start=}\n{c.end=}\n{c.text=}")

