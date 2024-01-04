import pymongo
from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.core.mongodb import MongoDBRepository

from .entities import EmbeddingRecord, EmbeddingSimilarity


class EmbeddingRepository(MongoDBRepository[EmbeddingRecord]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, EmbeddingRecord)

    @property
    def collection(self) -> str:
        return "embeddings"

    async def get_by_type(self, splitter_type: SplitterType, embedder_type: EmbedderType) -> list[EmbeddingRecord]:
        return await self.get_many(
            splitter_type=splitter_type,
            embedder_type=embedder_type,
        )

    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        method: str = "euclidean",
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        """Returns a list of Embeddings and their similarities that match the query vector."""
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
                            "$and": [
                                {"splitter_type": splitter_type},
                                {"embedder_type": embedder_type},
                            ]
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

    async def prepare_collection_resources(self) -> None:
        await self.db[self.collection].create_index("portrait_id")
        await self.db[self.collection].create_index(
            [
                ("splitter_type", pymongo.DESCENDING),
                ("embedder_type", pymongo.DESCENDING),
            ]
        )
        await self.db[self.collection].create_index(
            [
                ("portrait_id", pymongo.DESCENDING),
                ("splitter_type", pymongo.DESCENDING),
                ("embedder_type", pymongo.DESCENDING),
            ],
            unique=True,
        )
