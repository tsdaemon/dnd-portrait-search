import abc

from InstructorEmbedding import INSTRUCTOR

from portrait_search.core.config import EmbedderType


class Embedder:
    @classmethod
    @abc.abstractmethod
    def embedder_type(cls) -> EmbedderType:
        raise NotImplementedError()

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError()


class InstructorEmbeddings(Embedder, abc.ABC):
    def __init__(self, instructions: str, model_name: str):
        self.instructions = instructions
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = INSTRUCTOR(self.model_name)
        pairs = [[self.instructions, text] for text in texts]
        return model.encode(pairs).tolist()  # type: ignore


class InstructorEmbeddingsLargePathfinderCharacterInstructions(InstructorEmbeddings):
    def __init__(self) -> None:
        super().__init__(
            "Represents a description of a Pathfinder character:",
            model_name="hkunlp/instructor-large",
        )

    @classmethod
    def embedder_type(cls) -> EmbedderType:
        return EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS


EMBEDDERS: dict[EmbedderType, type[Embedder]] = {
    t.embedder_type(): t for t in [InstructorEmbeddingsLargePathfinderCharacterInstructions]
}
