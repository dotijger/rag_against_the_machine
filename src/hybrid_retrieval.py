# reciprocal rank fusion (RRF)
# rank number =  1 / (n + rank number) where n is a RFF constant (smoothing)
# fetch_n > top_n so not to miss the top docs
# when top_n is only 3 (k = 3)


def rrf(bm25_ranks: list[int], semantic_ranks: list[int]) -> list[int]:
    rrf_k = 60
    rrf_scores: dict[int, float] = {}

    for results in [bm25_ranks, semantic_ranks]:
        for rank, doc_idx in enumerate(results, start=1):
            score = 1.0 / (rrf_k + rank)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + score

    top_docs = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)

    return top_docs
