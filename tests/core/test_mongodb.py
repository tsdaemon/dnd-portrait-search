import pytest
from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.core.config import Config
from portrait_search.core.mongodb import MongoDBRecord, MongoDBRepository, get_connection


class FakeRecord(MongoDBRecord):
    some_field: str


class FakeRepository(MongoDBRepository[FakeRecord]):
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        super().__init__(db, FakeRecord)

    @property
    def collection(self) -> str:
        return "fake_collection"


@pytest.fixture
def fake_repository(mongodb_database_for_test: str) -> FakeRepository:
    return FakeRepository(get_connection(Config().mongodb_uri)[mongodb_database_for_test])


async def test_insert_get_delete(fake_repository: FakeRepository) -> None:
    fake_record = FakeRecord(some_field="some_value")
    new_fake_record = await fake_repository.insert(fake_record)
    assert new_fake_record.id is not None
    get_fake_record = await fake_repository.get(new_fake_record.id)
    assert get_fake_record.some_field == fake_record.some_field
    await fake_repository.delete(new_fake_record.id)
