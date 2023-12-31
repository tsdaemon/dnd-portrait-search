from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from .entities import PortraitRecord

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

    async def insert(self, portrait: PortraitRecord) -> PortraitRecord:
        entity = portrait.model_dump(by_alias=True, exclude={"id"})
        insertion_result = await self.db[COLLECTION_NAME].insert_one(entity)
        portrait.id = insertion_result.inserted_id
        return portrait

    async def get(self, id: ObjectId) -> PortraitRecord:
        entity = await self.db[COLLECTION_NAME].find_one({"_id": id})
        return PortraitRecord.model_validate(entity)
