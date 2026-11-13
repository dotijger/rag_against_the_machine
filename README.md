*This project has been created as part of the 42 curriculum by odschreu.*

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

https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
https://pydantic.dev/docs/validation/dev/concepts/models/
https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter
https://dev.to/klement_gunndu/10-chunking-strategies-that-make-or-break-your-rag-pipeline-4cng
https://www.geeksforgeeks.org/artificial-intelligence/chunking-strategies/
https://medium.com/data-science-collective/the-ultimate-text-chunking-toolkit-15-methods-with-python-code-9ef9d8f6a898
https://tqdm.github.io/
https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
https://www.youtube.com/watch?v=mxq4TkuVrWA
BOOK: Introduction to Information Retrieval (CH2, 6, 11)
