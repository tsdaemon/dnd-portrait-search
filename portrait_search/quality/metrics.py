import numpy as np

EXPECTED_QUERY_RESULT = list[tuple[str, float]]  # (path, relevance)
QUERY_RESULT_FOR_METRIC = tuple[list[str], EXPECTED_QUERY_RESULT]


def precision_at_k(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    """
    Calculate fraction of relevant results in the first k results.
    Only uses fully relevant results.
    """
    precisions = []
    for actual_result, expected_result_and_relevance in query_results:
        # In non-weighted metrics we do not count partially relevant results
        expected_result = [p for p, r in expected_result_and_relevance if r == 1.0]
        relevant_results = set(actual_result) & set(expected_result)
        precision = len(relevant_results) / len(actual_result)
        precisions.append(precision)
    return np.mean(precisions)  # type: ignore


def mean_reciprocal_rank(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    """
    Calculate Mean the average of the reciprocal ranks of results.
    Only uses fully relevant results.
    """
    reciprocal_ranks: list[float] = []
    for actual_result, expected_result_and_relevance in query_results:
        # In non-weighted metrics we do not count partially relevant results
        expected_result = [p for p, r in expected_result_and_relevance if r == 1.0]
        for i, p in enumerate(actual_result):
            if p in expected_result:
                reciprocal_ranks.append(1 / (i + 1))
                break
    return sum(reciprocal_ranks) / len(query_results)


def weighted_relevance_precision_at_k(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    """
    Calculate fraction of fully-relevant and partially relevant results in the first k results,
    weighted by the relevance.
    """
    precisions = []
    for actual_result, expected_result_and_relevance in query_results:
        expected_result, expected_relevance = zip(*expected_result_and_relevance)
        relevant_results = set(actual_result) & set(expected_result)
        precision = sum(expected_relevance[expected_result.index(p)] for p in relevant_results) / len(actual_result)
        precisions.append(precision)
    return np.mean(precisions)  # type: ignore


def weighted_relevance_mean_reciprocal_rank(query_results: list[QUERY_RESULT_FOR_METRIC]) -> float:
    """
    Calculate Mean the average of the reciprocal ranks of results, weighted by the relevance.
    """
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
    "mrr": mean_reciprocal_rank,
    "w_rel_precision@k": weighted_relevance_precision_at_k,
    "w_rel_mrr": weighted_relevance_mean_reciprocal_rank,
}
