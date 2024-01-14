import json
from collections.abc import Callable, Generator
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any

from portrait_search.core.enums import DistanceType, EmbedderType, SplitterType
from portrait_search.dependencies import Container
from portrait_search.embeddings.embedders import EMBEDDERS
from portrait_search.embeddings.splitters import SPLITTERS

from .judge import EvaluationResult, Judge

EXPERIMENT_TYPE = Callable[[Container], Judge]
EXPERIMENTS_COLLECTION_TYPE = dict[str, EXPERIMENT_TYPE]
PARAMETERS_GENERATOR_TYPE = Generator[tuple[EmbedderType, SplitterType, DistanceType, str], Any, Any]


def store_experiment_results(path: Path, results: dict[str, EvaluationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def load_or_create_experiment_results(path: Path) -> dict[str, EvaluationResult]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def all_possible_combinations_generator(experiment: str) -> PARAMETERS_GENERATOR_TYPE:
    for embedder, splitter, distance in product(EMBEDDERS, SPLITTERS, DistanceType):
        yield embedder, splitter, distance, experiment  # type: ignore


def all_possible_combinations_cosine_generator(experiment: str) -> PARAMETERS_GENERATOR_TYPE:
    for embedder, splitter in product(EMBEDDERS, SPLITTERS):
        yield embedder, splitter, DistanceType.COSINE, experiment  # type: ignore


def multi_experiment(
    generator: Callable[[str], PARAMETERS_GENERATOR_TYPE], experiment: str
) -> Generator[tuple[str, EXPERIMENT_TYPE], Any, Any]:
    for embedder, splitter, distance, dataset_name in generator(experiment):
        experiment_name = f"[{dataset_name}] Split {splitter}, embed {embedder}, distance {distance}"

        def generated_experiment(
            embedder: EmbedderType,
            splitter: SplitterType,
            distance: DistanceType,
            dataset_name: str,
            container: Container,
        ) -> Judge:
            with (
                container.embedder.override(EMBEDDERS[embedder]),
                container.splitter.override(SPLITTERS[splitter]),
                container.distance_type.override(distance),
            ):
                retriever = container.retriever()

            return Judge(retriever, dataset_name)

        yield experiment_name, partial(generated_experiment, embedder, splitter, distance, dataset_name)


def all_possible_combinations_by_experiment(experiment: str) -> EXPERIMENTS_COLLECTION_TYPE:
    return dict(multi_experiment(all_possible_combinations_generator, experiment))


def all_possible_combinations_cosine_by_experiment(experiment: str) -> EXPERIMENTS_COLLECTION_TYPE:
    return dict(multi_experiment(all_possible_combinations_cosine_generator, experiment))


CURRENT_EXPERIMENT: EXPERIMENTS_COLLECTION_TYPE = all_possible_combinations_cosine_by_experiment("v1")
