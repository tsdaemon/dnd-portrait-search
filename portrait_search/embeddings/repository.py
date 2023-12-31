from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.core.mongodb import MongoDBRepository

from .entities import EmbeddingRecord


class EmbeddingRepository(MongoDBRepository[EmbeddingRecord]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, EmbeddingRecord)
