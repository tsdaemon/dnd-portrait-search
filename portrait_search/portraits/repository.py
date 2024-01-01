from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.core.mongodb import MongoDBRepository

from .entities import PortraitRecord


class PortraitRepository(MongoDBRepository[PortraitRecord]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, PortraitRecord)

    @property
    def collection(self) -> str:
        return "portraits"

    async def get_distinct_hashes(self) -> set[str]:
        db_hashes = await self.db[self.collection].distinct("hash")
        return set(db_hashes)

    async def prepare_collection_resources(self) -> None:
        await self.db[self.collection].create_index("hash", unique=True)
        portrait_embeddings_pipeline = [
            {"$lookup": {"from": "embeddings", "localField": "id", "foreignField": "portrait_id", "as": "embeddings"}}
        ]
        await self.db.command(
            "create", "portrait_embeddings", viewOn=self.collection, pipeline=portrait_embeddings_pipeline
        )
