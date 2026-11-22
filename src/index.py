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
    def __init__(
        self, corpus: list[Chunk], loaded: bool, output_path: Path, mode: Mode
    ) -> None:
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
        with open(
            self.output_path / "chunks.jsonl", "w", encoding="utf-8"
        ) as f:
            for c in self.corpus:
                f.write(c.model_dump_json() + "\n")

    def index_corpus(self, mode: Mode) -> None:
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
