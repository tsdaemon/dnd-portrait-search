from typing import Any, Callable, Generator

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
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
