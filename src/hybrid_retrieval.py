# reciprocal rank fusion (RRF)
# rank number =  1 / (n + rank number) where n is the amount of documents FETCHED
# fetch_n > top_n as maybe the best canditate is on rank 4 in bm25 and rank 8 in semantic
# when top_n is only 3 (k = 3)

def rrf(fetch_k: int, bm25_ranks: list[int], semantic_ranks: list[int]) -> list[int]:
    rrf_scores: dict[int, float] = {}

    for results in [bm25_ranks, semantic_ranks]:
        for rank, doc_idx in enumerate(results, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (fetch_k + rank))
    
    top_docs = sorted(rrf_scores, key = lambda d: rrf_scores[d], reverse=True)

    return top_docs
