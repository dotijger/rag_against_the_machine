#!/usr/bin/env python3


# uv run python -m src <command> [options]
# necessary functionality / expected functionality of RAG:
# uv run python -m src index --max_chunk_size <int>
# uv run python -m src search <query> --k <int>
# uv run python -m src search_dataset --dataset_path <path> --k <int> --save_directory <dir>
# uv run python -m src answer <query> --k <int>
# uv run python -m src answer_dataset --student_search_results_path <path> --k <int> --save_directory <dir>
# uv run python -m src evaluate --student_search_results_path <path> --k <int> --dataset_path <dir>


import fire
import sys

# fire parses --max_chunk_size same as -m (1 letter abbrev with single dash) as well as leaving the flag out --
# it takes the first int it sees then as max_chunk_size
def index(max_chunk_size: int) -> str:
    return f"{max_chunk_size}"

def search(query: str, k: int) -> str:
    return f"{query}, {k}"

def help() -> str:
    return "help was called"

if __name__ == "__main__":
    fire.Fire()

