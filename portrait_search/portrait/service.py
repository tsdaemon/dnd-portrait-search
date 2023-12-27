from motor.motor_asyncio import AsyncIOMotorDatabase

COLLECTION_NAME = "portraits"


class PortraitService:
    def __init__(self, db: "AsyncIOMotorDatabase") -> None:
        self.db = db

    async def get_distinct_hashes(self) -> set[str]:
        db_hashes = await self.db[COLLECTION_NAME].distinct("hash")
        return set(db_hashes)

    async def prepare_indices(self) -> dict:
        hash_index = await self.db[COLLECTION_NAME].create_index("hash", unique=True)
        return {"hash": hash_index}
