import json
from collections.abc import Callable
from pathlib import Path

from portrait_search.core.enums import EmbedderType, SplitterType
from portrait_search.dependencies import Container
from portrait_search.embeddings.embedders import EMBEDDERS
from portrait_search.embeddings.splitters import SPLITTERS

from .judge import EvaluationResult, Judge


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


EXPERIMENTS: dict[str, Callable[[Container], Judge]] = {
    "[V1] 120 chunk and Instructor embeddings": experiment_v1_with_instructor_embeddings,
    "[V1] 120 chunk and all mini": experiment_v1_with_sent_trans_all_mini,
}
