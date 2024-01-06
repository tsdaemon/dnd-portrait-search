from portrait_search.portraits.entities import PortraitRecord
from portrait_search.retrieval.retriever import Retriever

from .dataset import load_dataset
from .metrics import METRICS

EvaluationResult = dict[str, float]


class Judge:
    def __init__(self, retriever: Retriever, experiment: str):
        self.retriever = retriever
        self.experiment = experiment

    async def evaluate(self) -> EvaluationResult:
        dataset = load_dataset(self.experiment)

        queries_results: list[tuple[str, str, list[PortraitRecord]]] = []

        for entry in dataset:
            for query in entry.queries:
                portraits, _ = await self.retriever.get_portraits(query, experiment=self.experiment)
                queries_results.append((entry.path, query, portraits))

        results_for_metrics = [(path, [p.fulllength_path for p in portraits]) for path, _, portraits in queries_results]
        result = {}
        for metric, f in METRICS.items():
            result[metric] = f(results_for_metrics)

        return result
