import abc
from collections.abc import Sequence

from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.core.mongodb import MongoDBRepository

from .entities import EmbeddingRecord, EmbeddingSimilarity, MongoEmbeddingRecord


class EmbeddingRepository(abc.ABC):
    @abc.abstractmethod
    async def get_by_type(self, splitter_type: SplitterType, embedder_type: EmbedderType) -> Sequence[EmbeddingRecord]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        experiment: str | None = None,
        method: str = "euclidean",
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def insert_many(self, records: Sequence[EmbeddingRecord]) -> Sequence[EmbeddingRecord]:
        raise NotImplementedError()


class MongoEmbeddingRepository(MongoDBRepository[MongoEmbeddingRecord]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, MongoEmbeddingRecord)

    @property
    def collection(self) -> str:
        return "embeddings"

    async def get_by_type(self, splitter_type: SplitterType, embedder_type: EmbedderType) -> Sequence[EmbeddingRecord]:
        return await self.get_many(
            splitter_type=str(splitter_type),
            embedder_type=str(embedder_type),
        )

    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        experiment: str | None = None,
        method: str = "euclidean",
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        """Returns a list of Embeddings and their similarities that match the query vector."""
        filter = [
            {"splitter_type": splitter_type},
            {"embedder_type": embedder_type},
        ]
        if experiment:
            filter.append({"experiment": experiment})
        entities = self.db[self.collection].aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": f"portrait-embeddings-search-{method}",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": limit * 20,
                        "limit": limit,
                        "filter": {
                            "$and": filter  # for some reason, $and is required here
                        },
                    }
                },
                {
                    "$project": {
                        "portrait_id": 1,
                        "embedding": 1,
                        "embedded_text": 1,
                        "similarity": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
        )
        return [EmbeddingSimilarity.model_validate(entity) async for entity in entities]
