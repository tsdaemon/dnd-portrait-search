def precision_at_k(query_results: list[tuple[str, list[str]]]) -> float:
    n_records = len(query_results)
    n_records_with_relevant_response = sum(1 for path, portraits in query_results if any(p == path for p in portraits))
    return float(n_records_with_relevant_response) / float(n_records)


def mean_reciprocal_rank(query_results: list[tuple[str, list[str]]]) -> float:
    reciprocal_ranks: list[float] = []
    for path, portraits in query_results:
        for i, p in enumerate(portraits):
            if p == path:
                reciprocal_ranks.append(1 / (i + 1))
                break
    return sum(reciprocal_ranks) / len(query_results)


METRICS = {"precision@k": precision_at_k, "mean_reciprocal_rank": mean_reciprocal_rank}
