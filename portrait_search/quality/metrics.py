import numpy as np

EXPECTED_QUERY_RESULT = list[tuple[str, float]]  # (path, relevance)
QUERY_RESULT_FOR_METRIC = tuple[list[str], EXPECTED_QUERY_RESULT]
MRR_RELEVANCE_THRESHOLD = 0.6


def precision_at_k(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    precisions = []
    for actual_result, expected_result_and_relevance in query_results:
        expected_result, _ = zip(*expected_result_and_relevance)
        relevant_results = set(actual_result) & set(expected_result)
        precision = len(relevant_results) / len(actual_result)
        precisions.append(precision)
    return np.mean(precisions)  # type: ignore


def mean_reciprocal_rank(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    reciprocal_ranks: list[float] = []
    for actual_result, expected_result_and_relevance in query_results:
        expected_result = [p for p, r in expected_result_and_relevance if r >= MRR_RELEVANCE_THRESHOLD]
        for i, p in enumerate(actual_result):
            if p in expected_result:
                reciprocal_ranks.append(1 / (i + 1))
                break
    return sum(reciprocal_ranks) / len(query_results)


def weighted_precision_at_k(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    precisions = []
    for actual_result, expected_result_and_relevance in query_results:
        expected_result, expected_relevance = zip(*expected_result_and_relevance)
        relevant_results = set(actual_result) & set(expected_result)
        precision = sum(expected_relevance[expected_result.index(p)] for p in relevant_results) / len(actual_result)
        precisions.append(precision)
    return np.mean(precisions)  # type: ignore


def weighted_mean_reciprocal_rank(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    reciprocal_ranks: list[float] = []
    for actual_result, expected_result_and_relevance in query_results:
        expected_result, expected_relevance = zip(*expected_result_and_relevance)
        for i, p in enumerate(actual_result):
            if p in expected_result:
                reciprocal_ranks.append(expected_relevance[expected_result.index(p)] / (i + 1))
                break
    return sum(reciprocal_ranks) / len(query_results)


METRICS = {
    "precision@k": precision_at_k,
    "mean_reciprocal_rank": mean_reciprocal_rank,
    "weighted_precision@k": weighted_precision_at_k,
    "weighted_mean_reciprocal_rank": weighted_mean_reciprocal_rank,
}
