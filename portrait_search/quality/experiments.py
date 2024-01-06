from portrait_search.core import Config, EmbedderType, SplitterType
from portrait_search.dependencies import Container

from .judge import Judge


def experiment_v1_with_instructor_embeddings() -> Judge:
    container = Container()
    config = Config()
    config.embedder_type = EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS
    config.splitter_type = SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60
    container.config.override(config)

    retriever = container.retriever()
    return Judge(retriever, "v1")


EXPERIMENTS = {
    "[V1] 120 chunk and Instructor embeddings": experiment_v1_with_instructor_embeddings,
}
