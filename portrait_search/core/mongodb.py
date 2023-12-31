import abc
from typing import Any, Callable, Generator, Generic, TypeVar

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import core_schema


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls) -> Generator[Callable[..., ObjectId], Any, Any]:
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> ObjectId:
        """Validates if the provided value is a valid ObjectId."""
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> core_schema.CoreSchema:
        """
        Defines the core schema for FastAPI documentation.
        Creates a JSON schema representation compatible with Pydantic's requirements.
        """
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(), python_schema=core_schema.is_instance_schema(ObjectId)
        )


def get_connection(url: str) -> AsyncIOMotorClient:
    return AsyncIOMotorClient(url)


def get_database(connection: AsyncIOMotorClient, database_name: str) -> AsyncIOMotorDatabase:
    return connection[database_name]


class MongoDBRecord(BaseModel):
    id: PyObjectId | None = Field(alias="_id", default=None)

    model_config = ConfigDict(populate_by_name=True)


TRecord = TypeVar("TRecord", bound=MongoDBRecord)


class MongoDBRepository(abc.ABC, Generic[TRecord]):
    def __init__(self, db: AsyncIOMotorDatabase, t: type[TRecord]) -> None:
        self.db = db
        self.t = t

    @abc.abstractproperty
    def collection(self) -> str:
        raise NotImplementedError()

    async def insert(self, record: TRecord) -> TRecord:
        entity = record.model_dump(by_alias=True, exclude={"id"})
        insertion_result = await self.db[self.collection].insert_one(entity)
        new_record = record.model_copy()
        new_record.id = insertion_result.inserted_id
        return new_record

    async def get(self, id: ObjectId) -> TRecord:
        entity = await self.db[self.collection].find_one({"_id": id})
        return self.t.model_validate(entity)

    async def delete(self, id: ObjectId) -> None:
        await self.db[self.collection].delete_one({"_id": id})
