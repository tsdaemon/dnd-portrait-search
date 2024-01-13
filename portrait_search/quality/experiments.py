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


def experiment_v1_with_instructor_embeddings(container: Container) -> Judge:
    with (
        container.embedder.override(EMBEDDERS[EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS]()),
        container.splitter.override(SPLITTERS[SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60]()),
    ):
        retriever = container.retriever()

    return Judge(retriever, "v1")


def experiment_v1_with_sent_trans_all_mini(container: Container) -> Judge:
    with (
        container.embedder.override(EMBEDDERS[EmbedderType.ALL_MINI_LM_L6_V2]()),
        container.splitter.override(SPLITTERS[SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60]()),
    ):
        retriever = container.retriever()

    return Judge(retriever, "v1")


def experiment_v1_with_sent_trans_msmarko(container: Container) -> Judge:
    with (
        container.embedder.override(EMBEDDERS[EmbedderType.MS_MARCO_DISTILBERT_BASE_V4]()),
        container.splitter.override(SPLITTERS[SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60]()),
    ):
        retriever = container.retriever()

    return Judge(retriever, "v1")


def experiment_v1_with_sent_trans_msmarko_cosine(container: Container) -> Judge:
    with (
        container.embedder.override(EMBEDDERS[EmbedderType.MS_MARCO_DISTILBERT_BASE_V4]()),
        container.splitter.override(SPLITTERS[SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60]()),
        container.distance_type.override(DistanceType.COSINE),
    ):
        retriever = container.retriever()

    return Judge(retriever, "v1")


def experiment_v1_with_sent_trans_msmarko_roberta_dot_product(container: Container) -> Judge:
    with (
        container.embedder.override(EMBEDDERS[EmbedderType.MS_MARCO_ROBERTA_BASE_ANCE_FIRSTP]()),
        container.splitter.override(SPLITTERS[SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60]()),
        container.distance_type.override(DistanceType.DOT_PRODUCT),
    ):
        retriever = container.retriever()

    return Judge(retriever, "v1")


def experiment_v1_with_sent_trans_all_mini_160(container: Container) -> Judge:
    with (
        container.embedder.override(EMBEDDERS[EmbedderType.ALL_MINI_LM_L6_V2]()),
        container.splitter.override(
            SPLITTERS[SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_160_OVERLAP_40_SAME_QUERY]()
        ),
    ):
        retriever = container.retriever()

    return Judge(retriever, "v1")


EXPERIMENTS: EXPERIMENTS_COLLECTION_TYPE = {
    "[V1] 120 chunk and Instructor embeddings": experiment_v1_with_instructor_embeddings,
    "[V1] 120 chunk and all mini": experiment_v1_with_sent_trans_all_mini,
    "[V1] 120 chunk and msmarko": experiment_v1_with_sent_trans_msmarko,
    "[V1] 120 chunk and msmarko, cosine": experiment_v1_with_sent_trans_msmarko_cosine,
    "[V1] 120 chunk and msmarko roberta, dot product": experiment_v1_with_sent_trans_msmarko_roberta_dot_product,
    "[V1] 160 chunk, same query and all mini": experiment_v1_with_sent_trans_all_mini_160,
}


def all_possible_combinations_generator(experiment: str) -> PARAMETERS_GENERATOR_TYPE:
    for embedder, splitter, distance in product(EmbedderType, SplitterType, DistanceType):
        yield embedder, splitter, distance, experiment  # type: ignore


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
                container.embedder.override(EMBEDDERS[embedder]()),
                container.splitter.override(SPLITTERS[splitter]()),
                container.distance_type.override(distance),
            ):
                retriever = container.retriever()

            return Judge(retriever, dataset_name)

        yield experiment_name, partial(generated_experiment, embedder, splitter, distance, dataset_name)


def all_possible_combinations_by_experiment(experiment: str) -> EXPERIMENTS_COLLECTION_TYPE:
    return dict(multi_experiment(all_possible_combinations_generator, experiment))
