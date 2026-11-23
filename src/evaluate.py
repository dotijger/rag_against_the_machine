import logging

from src.error import ImporterError
from src.import_utils import load_model
from src.models import (
    StudentSearchResults,
    Evaluation,
    MinimalSource,
    MinimalSearchResults,
    AnsweredQuestion,
    AnsweredRagDataset,
)

log = logging.getLogger(__name__)


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the intersection over union of two half-open ranges.

    The union is measured as the span from the earliest start to the
    latest end, so for two disjoint ranges it also covers the gap between
    them (the result is 0 in that case anyway).

    Args:
        a: (start, end) of the first range.
        b: (start, end) of the second range.

    Returns:
        A value in [0, 1]; 0.0 if the union is empty.
    """
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def matching_sources(result: MinimalSource, reference: MinimalSource) -> bool:
    """Decide whether a retrieved source matches a reference source.

    Args:
        result: A source returned by search.
        reference: A source from the reference dataset.

    Returns:
        True if both are in the same file and their ranges have an IoU of
        at least 0.05.
    """
    if result.file_path != reference.file_path:
        return False
    if (
        iou(
            (result.first_character_index, result.last_character_index),
            (reference.first_character_index, reference.last_character_index),
        )
        < 0.05
    ):
        return False
    return True


def get_student_search_dict(
    student_results: list[MinimalSearchResults],
) -> dict[str, list[MinimalSource]]:
    """Index search results by question ID.

    Args:
        student_results: One search result per question.

    Returns:
        A mapping from question ID to that question's retrieved sources.
    """
    return {s.question_id: s.retrieved_sources for s in student_results}


def get_recall_score(
    retrieved: list[MinimalSource], references: list[MinimalSource]
) -> float:
    """Compute recall for a single question.

    Args:
        retrieved: The sources retrieved for the question.
        references: The sources that should have been retrieved.

    Returns:
        The fraction of reference sources matched by at least one
        retrieved source, or 0.0 if there are no references.
    """
    references_matched = 0
    for reference in references:
        for source in retrieved:
            if matching_sources(source, reference):
                references_matched += 1
                break
    if references:
        return references_matched / len(references)
    else:
        return 0.0


def get_recall_k(
    k: int,
    student_results: dict[str, list[MinimalSource]],
    rag_questions: list[AnsweredQuestion],
) -> float:
    """Compute mean recall@k over a set of questions.

    Only the first k retrieved sources of each question are used.
    Questions with no search result, or an empty one, score 0 but still
    count towards the average.

    Args:
        k: How many of the top retrieved sources to consider.
        student_results: Retrieved sources keyed by question ID.
        rag_questions: The reference questions and their sources.

    Returns:
        Mean recall@k across all reference questions.
    """
    scores = 0.0
    for question in rag_questions:
        question_id, references = question.question_id, question.sources
        retrieved = student_results.get(question_id)
        if not retrieved:
            continue
        score = get_recall_score(retrieved[:k], references)
        scores += score
    return scores / len(rag_questions)


def run_evaluation(search_path: str, dataset_path: str) -> Evaluation:
    """Load search results and a reference dataset, then score them.

    Recall is computed for k in 1, 3, 5 and 10, stopping at the first k
    larger than the k the search results were produced with.

    Args:
        search_path: JSON file matching StudentSearchResults.
        dataset_path: JSON file matching AnsweredRagDataset.

    Returns:
        The recall@k metrics with the search k and question count.

    Raises:
        ImporterError: If either file cannot be read or validated.
    """
    range_k = [1, 3, 5, 10]
    try:
        student_results = load_model(search_path, StudentSearchResults)
        log.info("Student data is valid: True")
    except ImporterError as e:
        log.warning("Student data is valid: False")
        raise ImporterError(e) from e
    try:
        true_sources = load_model(dataset_path, AnsweredRagDataset)
    except ImporterError as e:
        raise ImporterError(e) from e

    metrics = []
    student_k = student_results.k
    student_dict = get_student_search_dict(student_results.search_results)
    for k in range_k:
        if k > student_k:
            break
        metrics.append(
            (k, get_recall_k(k, student_dict, true_sources.rag_questions))
        )

    return Evaluation(
        metrics=metrics, k=student_k, count=len(true_sources.rag_questions)
    )
