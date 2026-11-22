*This project has been created as part of the 42 curriculum by odschreu.*

# RAG against the machine

## Description

A retrieval-augmented generation (RAG) system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) codebase. It splits the repository's
Python files and Markdown docs into chunks, builds a lexical BM25 index over them,
retrieves the top-k chunks for a question and passes them as context to a local
**Qwen3-0.6B** model, which writes the answer.

Every retrieved source is reported as a `(file_path, first_character_index,
last_character_index)` span, so results can be scored automatically with
recall@k against ground-truth datasets.

The mandatory part is purely lexical (BM25). As a bonus, the system also
supports dense retrieval with `all-MiniLM-L6-v2` embeddings and hybrid retrieval
that merges both rankings with Reciprocal Rank Fusion.

## Instructions

### Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- About 2 GB of free disk space for the model weights (Qwen3-0.6B, plus
  MiniLM if you use semantic or hybrid mode)

On Codam machines, put the uv cache, the virtual environment and the Hugging Face
cache on goinfre so they don't fill your home quota:

```bash
export UV_CACHE_DIR=/goinfre/$USER/.uv-cache
export HF_HOME=/goinfre/$USER/.hf
```

### Installation

```bash
git clone <repo-url> rag_against_the_machine
cd rag_against_the_machine
make install        # or: uv sync
```

### Data

Copy (or clone) the vLLM repository into `data/raw/`:

```bash
git clone --depth 1 https://github.com/vllm-project/vllm data/raw
```

### Makefile

| Rule              | What it does                                   |
|-------------------|------------------------------------------------|
| `make install`    | Install dependencies with uv                   |
| `make run`        | Run the program, passing `args` through        |
| `make debug`      | Run under pdb, passing `args` through          |
| `make clean`      | Remove caches (`__pycache__`, `.mypy_cache`) and the log file |
| `make lint`       | Run `flake8 .` and `mypy .` with the subject's flags |
| `make lint-strict`| Run `flake8 .` and `mypy . --strict`           |

`make run` and `make debug` pass the `args` variable to the program, so these
two lines are equivalent:

```bash
make run args="search 'how is the KV cache sized' --k 5"
uv run python -m src search 'how is the KV cache sized' --k 5
```

Without `args`, `make run` prints the list of available commands.

### Commands

All commands run through the Fire CLI:

```bash
uv run python -m src <command> [options]
```

| Command          | Arguments                                                        | Output |
|------------------|------------------------------------------------------------------|--------|
| `index`          | `--max_chunk_size <int>` (1–2000, default 2000)                  | Builds the index in the processed directory |
| `search`         | `<query> --k <int>`                                              | Top-k sources for one query |
| `search_dataset` | `--dataset_path <path> --k <int> --save_directory <dir>`         | `StudentSearchResults` JSON |
| `answer`         | `<query> --k <int>`                                              | Retrieves, then answers one query |
| `answer_dataset` | `--student_search_results_path <path> --save_directory <dir>`    | `StudentSearchResultsAndAnswer` JSON |
| `evaluate`       | `--student_search_results_path <path> --dataset_path <path>`     | recall@1/3/5/10 |

`index`, `search`, `search_dataset` and `answer` also take `--mode`
(`bm25`, `semantic` or `hybrid`, default `bm25`). Semantic and hybrid search
read stored embeddings, so they need an index built in `semantic` or `hybrid`
mode. Both of those modes write the same files (BM25 index plus embeddings), so
one such index serves both search modes.

`evaluate` reports recall only up to the `k` stored in the results file, so run
`search_dataset` with `--k 10` to get all four values.

## Configuration

By default, the RAG system reads the source files from `data/raw` and writes
the index to `data/processed`, both relative to where you run the command.
You don't have to pass anything if that layout works for you.

To use other directories, pass `--raw_dir` and/or `--processed_dir`
**before** the command name:

```bash
# defaults: data/raw -> data/processed
uv run python -m src index --max_chunk_size 2000

# custom locations
uv run python -m src --raw_dir ~/repos/vllm --processed_dir /tmp/vllm_index index --max_chunk_size 2000
```

Commands that use the index (`search`, `search_dataset`, …) must be given the
same `--processed_dir` used when indexing, or they won't find it:

```bash
uv run python -m src --processed_dir /tmp/vllm_index search "how is the KV cache sized" --k 5
```

| Flag              | Default          | Used by                   |
|-------------------|------------------|---------------------------|
| `--raw_dir`       | `data/raw`       | `index`                   |
| `--processed_dir` | `data/processed` | `index` and all searches  |

Reported file paths start with the raw directory as you pass it. The public
datasets use the default layout, so evaluate with the default `--raw_dir`;
a different raw directory produces paths that do not match the ground truth.

Logs are shown on stderr (INFO and above) and written in full to `data/log`.

## Example usage

```bash
# 1. Build the index
uv run python -m src index --max_chunk_size 2000

# 2. Search a single question
uv run python -m src search "How does vLLM schedule requests?" --k 5

# 3. Answer a single question
uv run python -m src answer "How do I enable LoRA adapters?" --k 5

# 4. Search a whole dataset (scope the output folder by dataset, the public
#    datasets share file names)
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions

# 5. Measure recall@k against the ground truth
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 6. Generate answers from the saved search results
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture

The project has two phases that run as separate invocations. Indexing writes
everything to disk once; every later command loads it back.

```
                 index                                   search / answer
 data/raw ──► Chunker ──► Tokenizer ──► BM25 ──┐     query ──► Tokenizer ──► BM25 ──► top-k
                 │                              │                                       │
                 ▼                              ▼                                       ▼
          chunks.jsonl                    bm25s/ index          chunks.jsonl ──► context ──► Qwen3-0.6B ──► answer
                                        (embeddings.npy)
```

```
src/
├── __main__.py          Entry point: runs Fire, turns RAGError into a clean message
├── cli.py               Thin CLI layer: argument checks and terminal output
├── rag.py               RAGMachine, the core: index, search, answer, *_dataset
├── chunker.py           File-type-aware chunking (.py, .md, fallback window)
├── tokenizer.py         Identifier splitting, stopwords and stemming for BM25
├── index.py             Indexer: BM25 build/load, optional embeddings, retrieval
├── hybrid_retrieval.py  Reciprocal Rank Fusion of two rankings
├── transformer.py       AnswerGenerator: prompt building and Qwen3 generation
├── evaluate.py          recall@k against a ground-truth dataset
├── models.py            Pydantic models (subject models plus Chunk)
├── error.py             RAGError and its subclasses
├── import_utils.py      JSON loading and file helpers
└── log_setup.py         Rich console logging and the log file
```

The core (`RAGMachine`) never prints; it returns Pydantic models. The CLI
decides how to display them. This keeps the core reusable from tests or an
HTTP layer, and keeps the subject's JSON formats untouched.

The processed directory contains:

| File            | Contents |
|-----------------|----------|
| `chunks.jsonl`  | One `Chunk` per line: file path, start, end and text |
| `bm25s/`        | The saved `bm25s` index |
| `embeddings.npy`| MiniLM vectors, only in semantic and hybrid mode |

## Chunking strategy

Code and prose break naturally in different places, so each file type gets its
own chunker. Every chunker produces spans within `--max_chunk_size`
characters. The size must be between 1 and 2000 (the default): the moulinette
rejects any retrieved source longer than 2000 characters, so larger values are
refused before indexing starts.

- **Python (`.py`)**: candidate cuts are placed at lines that start a top-level
  `def`, `async def` or `class`. Decorators stay attached to the definition below
  them. Neighbouring small sections are then merged as long as they fit in the
  maximum size, and a section that is still too large is split at the last line
  break before the limit.
- **Markdown (`.md`)**: cuts at headings, ignoring `#` lines inside code fences.
  Small sections are merged in the same way, and oversized sections are split at
  line breaks.
- **Everything else (`.txt`)**: a fixed-size window.

Offsets are **character** indices, not bytes, and `last_character_index` is
exclusive, like a Python slice. The `Chunk` model validates every chunk on
creation: the start must be before the end and the text length must equal the
span length.

Chunk size matters more than it seems, because a retrieved span only counts as a
match if its IoU with the reference span is at least 0.05. A short reference
span inside a very large chunk can fail that test even though the right code
was found, so the maximum size is a ceiling, not a target.

## Retrieval method

Retrieval is BM25 via [`bm25s`](https://github.com/xhluca/bm25s). Most of the
gain came from the tokenizer, not from the scoring function:

- **Identifier splitting.** `snake_case` and `camelCase` identifiers are indexed
  both as the whole identifier and as their parts: `allocate_kv_cache` also gives
  `allocate`, `kv`, `cache`, and `CacheManager` gives `cache`, `manager`. A
  plain-English question can then match code it never quotes.
- **Lowercasing, English stopword removal and stemming**, applied identically to
  chunks at index time and to queries at search time.

**Bonus modes.** `semantic` embeds every chunk with `all-MiniLM-L6-v2` and ranks
by cosine similarity (normalized vectors, a single matrix product with numpy).
`hybrid` retrieves the top 60 candidates from both BM25 and the embeddings and
merges the two rankings with Reciprocal Rank Fusion (constant 60).

## Answer generation

`answer` runs a search, then generates. `answer_dataset` does **not** search
again: it reads an existing `StudentSearchResults` file, looks up the text of
each source in `chunks.jsonl` by its `(file, start, end)` key and generates from
that. This is why it takes no `k`; the value is inherited from the input file.

The prompt is built with Qwen3's own chat template. The fixed rule (answer using
only the provided context) goes in the system message; the retrieved chunks,
each labelled with its file path, and the question go in the user message.
Thinking is disabled with `enable_thinking=False`, decoding is greedy for
reproducibility, and only the newly generated tokens are decoded. The model is
loaded lazily, once per run, so `index` and `search` never load it.

## Performance analysis

Measured with `evaluate` on the public datasets (99 code questions, 100 docs
questions), `--max_chunk_size 2000`, k = 10.

**Code dataset**

| Mode                        | R@1   | R@3   | R@5   | R@10  |
|-----------------------------|-------|-------|-------|-------|
| Final BM25                  | 0.404 | 0.667 | 0.727 | 0.808 |
| Semantic (MiniLM)           | 0.212 | 0.354 | 0.374 | 0.495 |
| Hybrid (BM25 + MiniLM, RRF) | 0.364 | 0.515 | 0.586 | 0.737 |

**Docs dataset**

| Mode                        | R@1   | R@3   | R@5   | R@10  |
|-----------------------------|-------|-------|-------|-------|
| Final BM25                  | 0.620 | 0.790 | 0.850 | 0.880 |
| Semantic (MiniLM)           | 0.350 | 0.500 | 0.620 | 0.670 |
| Hybrid (BM25 + MiniLM, RRF) | 0.520 | 0.730 | 0.750 | 0.840 |

Observations:

- BM25 is the best mode on both datasets at every k, and the only one above
  both subject targets (80% recall@5 on docs, 50% on code).
- Semantic retrieval alone fails the code target (37% recall@5). MiniLM is
  trained on natural language and does not understand identifiers such as
  `allocate_kv_cache`. It does relatively better on docs (62%), which are prose.
- Hybrid retrieval scores below BM25 on both datasets. RRF weights both rankings
  equally, so the much weaker semantic ranking pushes good BM25 hits down. BM25
  therefore stays the default mode.
- Semantic and hybrid search are about 160x slower per query (~8 s vs ~0.05 s
  for 100 questions) because each query is embedded on the CPU.

Speed against the subject's limits (5 minutes to index, 90 seconds for 200
questions), measured on a CPU-only machine:

| Mode              | Indexing time | Retrieval, 199 questions |
|-------------------|---------------|--------------------------|
| BM25              | ~3.5 s        | ~0.1 s                   |
| Semantic / hybrid | ~200 s        | ~16 s                    |

Embedding all 13,502 chunks with MiniLM takes most of the hybrid indexing
time; it stays within the 5-minute limit, but with less margin than BM25.
- Docs has the smaller margin over its target (85% vs 80%), so it is the
  dataset most at risk on hidden questions.

## Design decisions

- **`bm25s` instead of `rank_bm25` or LangChain.** `bm25s` stores the index as
  sparse matrices and is much faster to build and query. LangChain would have
  added a large dependency for one function, and its defaults would hide the
  tokenizer, which is where most of the recall comes from.
- **`Chunk` subclasses `MinimalSource`.** A chunk is a source plus its text. The
  subclass passes anywhere a `MinimalSource` is expected, and results are always
  converted back to plain `MinimalSource` objects before they are saved, so the
  subject's output format is never changed.
- **Subject models are left as specified.** No extra fields (such as the
  retrieval mode) are added to `StudentSearchResults`; anything the program needs
  later is recovered from the index.
- **Directories as constructor flags.** The raw and processed directories are
  Fire constructor arguments with documented defaults, not hardcoded paths.
- **Load once, lazily.** The index, chunks and model are loaded the first time a
  command needs them and reused, never once per question.
- **One error type at the boundary.** All expected failures raise a subclass of
  `RAGError`, and `__main__` turns them into a one-line message instead of a
  traceback.

## Challenges faced

- **One shared vocabulary.** Calling `bm25s.tokenize` separately for each chunk
  gave every chunk its own token-ID vocabulary, so BM25 returned
  plausible-looking but meaningless scores. The fix was to have the tokenizer
  return plain string tokens and let `bm25s` build a single vocabulary over the
  whole corpus.
- **Character vs byte offsets.** `ast` reports column offsets in bytes, which
  silently breaks character spans on any line with non-ASCII text. The fix was
  line-based chunking on the decoded text, with the `Chunk` validator checking
  that every text matches its span length.
- **IoU and large chunks.** Early on the right file was often retrieved but
  scored as a miss because the chunk was much larger than the reference span.
- **Fire's quirks.** Fire prints whatever a command returns (whole Pydantic
  objects at first), guesses argument types (`--k abc` arrives as a string), and
  requires constructor flags before the command name. Commands now return
  nothing and print through the CLI layer, and numeric arguments are checked
  before use.
- **Missing files.** `glob("*")` instead of `rglob("*")` silently skipped every
  subdirectory of the repository until the chunk count looked suspiciously low.
- **Answering from saved results.** `answer_dataset` only receives spans, not
  text, so the text has to be looked up in the current index. If the index is
  rebuilt with another chunk size in between, the spans no longer match; this
  is caught and reported, asking you to re-run `search_dataset`.

## Resources

**Retrieval and BM25**

- *Introduction to Information Retrieval*, Manning, Raghavan and Schütze:
  chapter 2 (tokenization and the term vocabulary), chapter 6 (term weighting
  and the vector space model), chapter 11 (probabilistic IR and BM25)
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf),
  Robertson and Zaragoza, the reference paper for BM25
- [BM25 explained (video)](https://www.youtube.com/watch?v=mxq4TkuVrWA)

**Chunking**

- [LangChain `RecursiveCharacterTextSplitter`](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter),
  as a reference for recursive splitting
- [10 chunking strategies that make or break your RAG pipeline](https://dev.to/klement_gunndu/10-chunking-strategies-that-make-or-break-your-rag-pipeline-4cng)
- [Chunking strategies (GeeksforGeeks)](https://www.geeksforgeeks.org/artificial-intelligence/chunking-strategies/)
- [The ultimate text chunking toolkit: 15 methods with Python code](https://medium.com/data-science-collective/the-ultimate-text-chunking-toolkit-15-methods-with-python-code-9ef9d8f6a898)

**Models**

- [Qwen concepts: chat templates, tokens and thinking mode](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [sentence-transformers/all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [all-MiniLM-L6-v2 explained](https://levelup.gitconnected.com/all-minilm-l6-v2-358e826ef499)

**Tooling**

- [Pydantic: models](https://pydantic.dev/docs/validation/dev/concepts/models/)
- [tqdm](https://tqdm.github.io/)

### How AI was used

I used Claude (Anthropic) as a tutor and reviewer throughout the project:

- **Concepts:** explanations of BM25, TF-IDF, inverted indexes, recall@k and IoU
  matching, and Qwen3's chat template and special tokens, usually followed by
  reading the sources above.
- **Debugging:** finding the cause of bugs such as the per-chunk tokenization
  mismatch, the `glob`/`rglob` mistake and Fire's flag ordering.
- **Design discussion:** weighing options (for example `bm25s` vs `rank_bm25`,
  how to structure `Chunk`, whether to add ChromaDB) before deciding myself.
- **Review:** running flake8 and mypy over the code and pointing out bugs and
  unmet requirements.
- **Documentation:** drafting this README, which I then checked and edited.

All code was written, tested and understood by me; every design choice above is
one I can explain and defend.
