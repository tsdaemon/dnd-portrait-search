from collections import defaultdict

import numpy as np

from portrait_search.retrieval.retriever import Retriever

from .dataset import DatasetEntry, load_dataset
from .metrics import EXPECTED_QUERY_RESULT, METRICS

EvaluationResult = dict[str, float]


def prepare_expected_results(dataset: list[DatasetEntry]) -> dict[str, EXPECTED_QUERY_RESULT]:
    # First prepare vocabulary to estimate how often each match is used
    vocabulary: dict[str, int] = defaultdict(int)
    for entry in dataset:
        for query in entry.queries:
            for match in query.match:
                vocabulary[match] += 1

    # Next calculate relevance for each expected result
    expected_results: dict[str, EXPECTED_QUERY_RESULT] = {}
    for entry in dataset:
        for query in entry.queries:
            expected_results[query.query] = []
            relevances: list[float] = [1 / vocabulary[match] for match in query.match]
            relevances = (np.array(relevances) / np.sum(relevances)).tolist()
            mataches_relevances = dict(zip(query.match, relevances))

            for portrait in query.portraits:
                portrait_relevance: float = sum((mataches_relevances[match] for match in portrait.match), 0.0)
                expected_results[query.query].append((portrait.path, portrait_relevance))

    return expected_results


class Judge:
    def __init__(self, retriever: Retriever, experiment: str):
        self.retriever = retriever
        self.experiment = experiment

    async def evaluate(self) -> EvaluationResult:
        dataset = load_dataset(self.experiment)
        query_to_expected_results = prepare_expected_results(dataset)

        results_for_metrics: list[tuple[list[str], EXPECTED_QUERY_RESULT]] = []

        for query, expected_result in query_to_expected_results.items():
            portraits, _ = await self.retriever.get_portraits(query, experiment=self.experiment)
            results_for_metrics.append(([p.fulllength_path for p in portraits], expected_result))

        result = {}
        for metric, f in METRICS.items():
            result[metric] = f(results_for_metrics)

        return result
