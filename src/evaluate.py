from src.error import RAGError
from src.import import load_model
from src.models import StudentSearchResults, Evaluation, MinimalSource, MinimalSearchResults, AnsweredRagDataset

def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

def matching_sources(result: MinimalSource, reference: MinimalSource) -> bool:
    if result.file_path != reference.file_path:
        return False
    if iou((result.first_character_index, result.last_character_index),
           (reference.first_character_index, reference.last_character_index)) < 0.05:
        return False
    return True
        
def get_student_search_dict(student_results: list[MinimalSearchResults]) -> dict[str, list[MinimalSource]]:
    return {s.question_id: s.retrieved_sources for s in student_results}

def get_recall_score(retrieved: list[MinimalSource], references: list[MinimalSource]) -> float:
    references_matched = 0
    for reference in references:
        for source in retrieved:
            if matching_sources(source, reference):
                references_matched += 1
                break

    return references_matched / len(references)

def get_recall_k(k: int, student_results: dict[str, list[MinimalSource]], rag_questions: list[AnsweredQuestion]) -> float:
    scores = 0.0
    for question in rag_questions:
        question_id, references = question.question_id, question.sources
        retrieved = student_results.get(question_id)
        if not retrieved:
            raise RAGError(f"No student search results for {question_id} in evaluation.")
        score = get_recall_score(retrieved[:k], references)
        scores += score
    return (scores / len(rag_questions))

def run_evaluation(search_path: str, dataset_path: str) -> Evaluation:
    range_k = [1, 3, 5, 10]
    student_results = load_model(search_path, StudentSearchResults)
    true_sources = load_model(dataset_path, AnsweredRagDataset)

    metrics = []
    student_k = student_results.k
    student_dict = get_student_search_dict(student_results.search_results)
    for k in range_k:
        if k > student_k:
            break
        metrics.append((k, get_recall_k(k, student_dict, true_sources.rag_questions)))
    
    return Evaluation(
        metrics=metrics,
        k=student_k,
        count=len(true_sources)
        )


# for historical purposes, scrapped:

def extract_student_search_results(path: str) -> StudentSearchResults:
    data = json_load(Path(path))
    try:
        search_dataset = StudentSearchResults.model_validate_json(Path(path).read_text(encoding='utf-8'))
    except ValidationError as e:
        raise RAGError(f"{path} does not have a valid Student Search Results structure.")
    if not isinstance(data['k'], int):
        raise ValueError(f"k has to be an positive integer, now {data['k']}")
    if data.get('k') <= 0:
        raise ValueError(f"k has to be a positive integer, now {data['k']}")
    if len(data.get('search_results')) == 0:
        raise RAGError('Search results are empty, nothing to evaluate.')
    return StudentSearchResults(
            search_results=data['search_results'],
            k=data['k']
            )
