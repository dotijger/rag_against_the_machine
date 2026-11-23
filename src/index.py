from pathlib import Path
import bm25s
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

from src.models import Chunk, Mode
from src.tokenizer import tokenize
from src.hybrid_retrieval import rrf
from src.error import IndexerError

log = logging.getLogger(__name__)


class Indexer:
    """Own the BM25 index and the embeddings for one chunk corpus.

    Positions in corpus are the shared document IDs: both
    retrievers return indices into this list.

    Attributes:
        corpus: The chunks being indexed, in a fixed order.
        output_path: Directory the index files are saved to and loaded
            from.
        bm25s_retriever: The lexical index.
        minilm_retriever: The embedding model; only present for the
            semantic and hybrid modes.
        minilm_embeddings: Array of shape (len(corpus), 384); only
            present once built or loaded for an embedding mode.
    """

    def __init__(
        self, corpus: list[Chunk], loaded: bool, output_path: Path, mode: Mode
    ) -> None:
        """Prepare a fresh index, or load an existing one from disk.

        The embedding model is only loaded for modes that need it. When
        building a fresh bm25 index, any old embeddings.npy is
        deleted so it cannot later be paired with the wrong chunks.

        Args:
            corpus: The chunks to index, or that the saved index was built
                from.
            loaded: True to load the saved index from output_path;
                False to start empty and call index_corpus.
            output_path: The processed-data directory.
            mode: Retrieval mode the index will be used with.

        Raises:
            IndexerError: If loading for an embedding mode and the
                embeddings are missing or their row count does not match
                the corpus.
        """
        self.corpus = corpus
        if mode != "bm25":
            self.minilm_retriever = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
        if not loaded:
            self.bm25s_retriever = bm25s.BM25()
            if mode == "bm25":
                (output_path / "embeddings.npy").unlink(missing_ok=True)
        else:
            self.bm25s_retriever = bm25s.BM25.load(output_path / "bm25s")
            if mode == "semantic" or mode == "hybrid":
                try:
                    self.minilm_embeddings = np.load(
                        output_path / "embeddings.npy"
                    )
                except OSError as e:
                    raise IndexerError(f"{e}") from e
                if self.minilm_embeddings.shape[0] != len(corpus):
                    raise IndexerError(
                        "Stale semantic index! Re-run index with semantic"
                        "or hybrid mode. Embeddings do not match chunks.jsonl"
                    )
        self.output_path = output_path

    def _save_chunk_lib(self) -> None:
        """Write the corpus to chunks.jsonl, one chunk per line."""
        with open(
            self.output_path / "chunks.jsonl", "w", encoding="utf-8"
        ) as f:
            for c in self.corpus:
                f.write(c.model_dump_json() + "\n")

    def index_corpus(self, mode: Mode) -> None:
        """Build the indexes for the corpus and save everything to disk.

        The BM25 index is always built. Embeddings are computed and saved
        as well for the semantic and hybrid modes.

        Args:
            mode: bm25, semantic or hybrid.

        Raises:
            IndexerError: If mode is not one of the above.
        """
        modes = ["semantic", "hybrid", "bm25"]
        if mode not in modes:
            raise IndexerError(
                f"Unknown mode {mode}, choose from: bm25, semantic, hybrid"
            )

        corpus_tokens = [
            tokenize(c.text)
            for c in tqdm(self.corpus, desc="Tokenizing corpus...")
        ]

        if mode == "semantic" or mode == "hybrid":
            texts = [c.text for c in self.corpus]
            self.minilm_embeddings = self.minilm_retriever.encode(
                texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True,
            )  # shape : (n_chunks, 384), dtype float32
            np.save(
                self.output_path / "embeddings.npy", self.minilm_embeddings
            )

        self.bm25s_retriever.index(corpus_tokens, show_progress=True)
        self.bm25s_retriever.save(self.output_path / "bm25s")

        self._save_chunk_lib()

    def _get_fetchk_idx(self, k: int, query: str, retriever: str) -> list[int]:
        """Return the top k chunk indices from a single retriever.

        For semantic, embeddings are normalised, so the dot product is
        the cosine similarity. argpartition finds the top k without
        a full sort, and only those k are then sorted.

        Args:
            k: Number of results; must be smaller than the corpus size
                for semantic.
            query: The question text.
            retriever: bm25 or semantic.

        Returns:
            Chunk indices, best match first.

        Raises:
            IndexerError: If retriever is not recognised.
        """
        if retriever == "bm25":
            ids, scores = self.bm25s_retriever.retrieve([tokenize(query)], k=k)
            return [int(i) for i in ids[0]]
        if retriever == "semantic":
            query_tensor = self.minilm_retriever.encode(
                query, normalize_embeddings=True
            )
            scores = self.minilm_embeddings @ query_tensor
            top = np.argpartition(-scores, k)[:k]
            top = top[np.argsort(-scores[top])]
            return [int(i) for i in top]
        raise IndexerError(
            f"Unknown mode {retriever}, choose from: bm25, semantic, hybrid"
        )

    def get_ktop_idx(self, query: str, k: int, mode: Mode) -> list[int]:
        """Return the top k chunk indices for a query.

        In hybrid mode each retriever fetches a deeper list (at least
        60, capped at the corpus size) before fusion, so a chunk ranked
        moderately by both can still reach the final top k.

        Args:
            query: The question text.
            k: Number of results to return.
            mode: bm25, semantic or hybrid.

        Returns:
            Chunk indices, best match first.

        Raises:
            IndexerError: If mode is not recognised.
        """
        if mode == "bm25":
            return self._get_fetchk_idx(k, query, "bm25")
        if mode == "semantic":
            return self._get_fetchk_idx(k, query, "semantic")
        if mode == "hybrid":
            fetch_k = min(max(60, k), len(self.corpus))
            bm25_ranks = self._get_fetchk_idx(fetch_k, query, "bm25")
            semantic_ranks = self._get_fetchk_idx(fetch_k, query, "semantic")
            return rrf(bm25_ranks, semantic_ranks)[:k]
        raise IndexerError(
            f"Unknown mode {mode}, choose from: bm25, semantic, hybrid"
        )
