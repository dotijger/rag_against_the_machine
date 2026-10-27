
class RAG():

    def index(max_chunk_size: int) -> None:
        print(f"{max_chunk_size=}")

    def search(query: str, k: int) -> None:
        print(f"{query}, k = {k}")

    def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
        print(f"{dataset_path=}, {k=}, {save_directory=}")

    def answer(query: str, k: int) -> None:
        print(f"{query}, k = {k}")

    def answer_dataset(student_search_results_path: str, save_directory: str) -> None:
        print(f"{student_search_results_path=}, {save_directory=}")

    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        print(f"{student_search_results_path=}, {dataset_path=}")
