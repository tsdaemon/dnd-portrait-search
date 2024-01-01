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
    new_fake_record = await fake_repository.insert_one(fake_record)
    assert new_fake_record.id is not None
    get_fake_record = await fake_repository.get_one(new_fake_record.id)
    assert get_fake_record.some_field == fake_record.some_field
    await fake_repository.delete(new_fake_record.id)


async def test_get_many(fake_repository: FakeRepository) -> None:
    fake_record1 = FakeRecord(some_field="some_value")
    new_fake_record1 = await fake_repository.insert_one(fake_record1)
    fake_record2 = FakeRecord(some_field="some_value")
    new_fake_record2 = await fake_repository.insert_one(fake_record2)
    fake_record3 = FakeRecord(some_field="some_other_value")
    new_fake_record3 = await fake_repository.insert_one(fake_record3)
    many_fake_records = await fake_repository.get_many(some_field="some_value")
    assert len(many_fake_records) == 2
    assert new_fake_record1 in many_fake_records
    assert new_fake_record2 in many_fake_records
    assert new_fake_record1.id is not None
    await fake_repository.delete(new_fake_record1.id)
    assert new_fake_record2.id is not None
    await fake_repository.delete(new_fake_record2.id)
    assert new_fake_record3.id is not None
    await fake_repository.delete(new_fake_record3.id)
