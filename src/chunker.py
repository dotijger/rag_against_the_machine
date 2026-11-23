# chunking strategies:
# 1 python code chunking (.py)
# 2 markdown / text chunking (.txt, .md)
# use of TD-IDF -> BM25 lexical retieving methods

from pydantic import ValidationError
from pathlib import Path
import re
import logging

from src.error import ChunkError
from src.models import Chunk

log = logging.getLogger(__name__)


class Chunker:
    """Chunk a single file according to its extension.

    Call `run` once; the result is left in self.chunks.

    Attributes:
        SUPPORTED: File extensions this chunker can handle.
        path: The file being chunked.
        max_chunk_size: Upper bound on a chunk's length, in characters.
        chunks: Chunks produced by `run`, in file order.
        body: The full text of the file once it has been read.
    """

    SUPPORTED = [".txt", ".md", ".py"]

    def __init__(self, path: Path, max_chunk_size: int = 2000) -> None:
        """Set up a chunker for one file without reading it yet.

        Args:
            path: The file to chunk.
            max_chunk_size: Upper bound on a chunk's length, in characters.
        """
        self.path = path
        self.max_chunk_size = max_chunk_size
        self.chunks: list[Chunk] = []
        self.body: str = ""

    def _read_source(self) -> None:
        """Read the file into self.body.

        ``newline=""`` keeps line endings exactly as they are on disk, so
        character offsets match the original file.
        """
        with open(self.path, encoding="utf-8", newline="") as f:
            self.body = f.read()

    def _window_chunker(self, start: int = 0, stop: int = -1) -> None:
        """Cut self.body[start:stop] into fixed-size, non-overlapping chunks.

        Used for plain-text files, which have no structure to cut on.

        Args:
            start: Offset of the first character to chunk.
            stop: Offset one past the last character to chunk; -1 means
                the end of the file.

        Raises:
            ChunkError: If a produced chunk fails validation.
        """
        step = self.max_chunk_size
        if stop == -1:
            stop = len(self.body)
        for idx in range(start, stop, step):
            try:
                self.chunks.append(
                    Chunk(
                        file_path=str(self.path),
                        first_character_index=idx,
                        last_character_index=min(idx + step, stop),
                        text=self.body[idx: idx + step],
                    )
                )
            except ValidationError as e:
                raise ChunkError(e)

    def run(self) -> None:
        """Read the file and chunk it with the strategy for its extension.

        Unsupported extensions are logged and skipped; empty files produce
        no chunks.

        Raises:
            ChunkError: If a produced chunk fails validation.
        """
        if self.path.suffix not in self.SUPPORTED:
            log.warning(f"Skipping '{self.path}', file type not supported.")
            return
        try:
            self._read_source()
        except (OSError, UnicodeDecodeError) as e:
            log.warning(f"Skipping {self.path}, could not read it: {e}.")
            return
        if len(self.body) == 0:
            return
        if self.path.suffix == ".txt":
            self._window_chunker()
        elif self.path.suffix == ".md":
            self._md()
        elif self.path.suffix == ".py":
            self._py()

    def _py(self) -> None:
        """Chunk Python source at top-level definitions.

        A cut is placed at the start of every unindented line beginning
        with ``def``, ``async def``, ``class`` or ``@``. A definition line
        directly after a decorator is not cut, so decorators stay with
        what they decorate. Cuts less than 100 characters after the
        previous one are skipped, which avoids tiny chunks.
        """
        starters = ("def", "async def", "class", "@")
        lines = self.body.split("\n")
        cuts: list[int] = []
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
        """Chunk Markdown at headings.

        A cut is placed at the start of every ATX heading (``#`` to
        ``######`` followed by whitespace). Lines inside fenced code blocks
        (````` or ``~~~``) are ignored, so a ``#`` comment in a code sample
        is not mistaken for a heading. Cuts less than 100 characters after
        the previous one are skipped.
        """
        heading = re.compile(r"#{1,6}\s")
        code_header = re.compile(r"(```|~~~)")
        lines = self.body.split("\n")
        cuts: list[int] = []
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
        """Split (start, end) into ranges no longer than the maximum.

        Each range ends just after the last newline that fits inside the
        size limit, so lines are kept whole where possible. If there is no
        newline in reach, the range is cut at exactly the limit.

        Args:
            start: Offset where the segment begins.
            end: Offset one past where the segment ends.

        Returns:
            Consecutive (start, end) ranges covering the whole segment.
        """
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
        """Drop cuts so neighbouring small segments are combined.

        Walks the cuts in order and keeps extending the current segment
        while it still fits in max_chunk_size. A segment that is too
        large on its own is kept as its own boundary and split later by
        self._windows().

        Args:
            cuts: Ascending end offsets of the structural segments; the
                last one must be len(body).

        Returns:
            Ascending end offsets of the merged segments.
        """
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
        """Turn segment boundaries into `Chunk` objects.

        Each segment (previous_cut, cut) is passed through
        self._windows(), so oversized segments become several chunks.

        Args:
            cuts: Ascending end offsets of the segments, starting from 0.

        Raises:
            ChunkError: If a produced chunk fails validation.
        """
        previous = 0
        for cut in cuts:
            for start, end in self._windows(previous, cut):
                try:
                    self.chunks.append(
                        Chunk(
                            file_path=str(self.path),
                            first_character_index=start,
                            last_character_index=end,
                            text=self.body[start:end],
                        )
                    )
                except ValidationError as e:
                    raise ChunkError(e)
            previous = cut
