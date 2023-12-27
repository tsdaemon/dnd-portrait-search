from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.portrait.entity import PortraitRecord

COLLECTION_NAME = "portraits"


class PortraitRepository:
    def __init__(self, db: "AsyncIOMotorDatabase") -> None:
        self.db = db

    async def get_distinct_hashes(self) -> set[str]:
        db_hashes = await self.db[COLLECTION_NAME].distinct("hash")
        return set(db_hashes)

    async def prepare_indices(self) -> dict:
        hash_index = await self.db[COLLECTION_NAME].create_index("hash", unique=True)
        return {"hash": hash_index}

    async def create(self, portrait: PortraitRecord) -> None:
        entity = portrait.model_dump(by_alias=True, exclude={"id"})
        await self.db[COLLECTION_NAME].insert_one(entity)
