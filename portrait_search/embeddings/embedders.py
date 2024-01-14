import abc
from typing import Any

import numpy as np
from InstructorEmbedding import INSTRUCTOR
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from portrait_search.core.enums import EmbedderType


class Embedder(abc.ABC):
    def __init__(self, expected_dimensionality: int | None = None) -> None:
        self.expected_dimensionality = expected_dimensionality
        self._type: EmbedderType | None = None

    def __repr__(self) -> str:
        return f"Embedder {self._type}"

    @property
    def type(self) -> EmbedderType:
        if self._type is None:
            raise ValueError("Embedder type not set")
        return self._type

    @type.setter
    def type(self, value: EmbedderType) -> None:
        self._type = value

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embed(texts)
        return self._match_dimensionality(vectors).tolist()

    @abc.abstractmethod
    def _embed(self, texts: list[str]) -> NDArray[np.float_]:
        raise NotImplementedError()

    def _match_dimensionality(self, vectors: NDArray[np.float_]) -> NDArray[np.float_]:
        if self.expected_dimensionality is None:
            return vectors

        if self.expected_dimensionality == vectors.shape[1]:
            return vectors

        if self.expected_dimensionality < vectors.shape[1]:
            raise ValueError(
                f"Expected dimensionality {self.expected_dimensionality} is smaller than the"
                f"actual dimensionality {vectors.shape[1]}. Can not reduce dimensionality."
            )

        zeros = np.zeros((vectors.shape[0], self.expected_dimensionality - vectors.shape[1]))
        result = np.concatenate((vectors, zeros), axis=1)
        return result


EMBEDDERS: dict[EmbedderType, Embedder] = {}


def register_embedder(embedder_type: EmbedderType, embedder: Embedder) -> None:
    EMBEDDERS[embedder_type] = embedder
    embedder.type = embedder_type


class InstructorEmbedder(Embedder):
    def __init__(self, instructions: str, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.instructions = instructions
        self.model = INSTRUCTOR(model_name)

    def _embed(self, texts: list[str]) -> NDArray[np.float_]:
        pairs = [[self.instructions, text] for text in texts]
        return np.array(self.model.encode(pairs))  # type: ignore


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = SentenceTransformer(model_name)

    def _embed(self, texts: list[str]) -> NDArray[np.float_]:
        return np.array(self.model.encode(texts, convert_to_numpy=True))


register_embedder(
    EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
    InstructorEmbedder("Represents a description of a Pathfinder character:", "hkunlp/instructor-large"),
)
# register_embedder(EmbedderType.ALL_MINI_LM_L6_V2, SentenceTransformerEmbedder("all-MiniLM-L6-v2"))
# register_embedder(EmbedderType.MS_MARCO_DISTILBERT_BASE_V4, SentenceTransformerEmbedder("msmarco-distilbert-base-v4"))
# register_embedder(
#     EmbedderType.MS_MARCO_ROBERTA_BASE_ANCE_FIRSTP, SentenceTransformerEmbedder("msmarco-roberta-base-ance-firstp")
# )
