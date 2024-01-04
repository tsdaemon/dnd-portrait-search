import abc

from portrait_search.embeddings import EmbeddingSimilarity
from portrait_search.portraits import PortraitRecord


class Retriever(abc.ABC):
    @abc.abstractmethod
    async def get_portraits(
        self, query: str, limit: int = 10
    ) -> tuple[list[PortraitRecord], list[list[EmbeddingSimilarity]]]:
        raise NotImplementedError()
