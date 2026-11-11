# retrieval

```bash
uv run python3 -m src index -max_chunk_size <int>

```

for INDEXING:
- chunker
    can still use LangChain RecursiveTextSplitter (and specify language python / md etc)
- indexer


```bash

uv run python3 -m src search <query> -k <int>

uv run python3 -m src search_dataset -dataset_path <path> -k <int> -save_directory <dir>

```
for SEARCHING:
- searcher based on indexed base


```bash

uv run python3 -m evaluate -student_search_results_path <path> -dataset_path <path>

```

for EVALUATION:
- evaluator that runs on the answered json and retrieves the 'correct' source per
question id, and then goes over the model's search results. recall@k is then the % (0-1)
that it has the correct source in the k amount of sources for that search result.

# generation


```bash

uv run python3 -m src answer <query> -k <int>

uv run python3 -m src answer_dataset -student_search_results_path <path> -save_directory <dir>

```
for ANSWERING:

for LLM :
- prompt generator
- answer generator



