import abc
from typing import Any

import numpy as np
from InstructorEmbedding import INSTRUCTOR
from numpy.typing import NDArray

from portrait_search.core.config import EmbedderType


class Embedder:
    def __init__(self, expected_dimensionality: int | None = None) -> None:
        self.expected_dimensionality = expected_dimensionality

    @classmethod
    @abc.abstractmethod
    def embedder_type(cls) -> EmbedderType:
        raise NotImplementedError()

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


class InstructorEmbeddings(Embedder, abc.ABC):
    def __init__(self, instructions: str, model_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.instructions = instructions
        self.model = INSTRUCTOR(model_name)

    def _embed(self, texts: list[str]) -> NDArray[np.float_]:
        pairs = [[self.instructions, text] for text in texts]
        return np.array(self.model.encode(pairs))  # type: ignore


class InstructorEmbeddingsLargePathfinderCharacterInstructions(InstructorEmbeddings):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            "Represents a description of a Pathfinder character:", "hkunlp/instructor-large", *args, **kwargs
        )

    @classmethod
    def embedder_type(cls) -> EmbedderType:
        return EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS


EMBEDDERS: dict[EmbedderType, type[Embedder]] = {
    t.embedder_type(): t for t in [InstructorEmbeddingsLargePathfinderCharacterInstructions]
}
