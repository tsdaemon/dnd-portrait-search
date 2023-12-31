import pytest

from portrait_search.dependencies import Container
from portrait_search.portraits.entities import PortraitRecord
from portrait_search.portraits.repository import PortraitRepository


@pytest.fixture
def portraits_repository_for_tests(container: Container) -> PortraitRepository:
    container.init_resources()
    return container.portrait_repository()


async def test_portrait_repository_insert_and_get(
    mongodb_database_for_test: str, portraits_repository_for_tests: PortraitRepository
) -> None:
    portrait_record = PortraitRecord(
        description="",
        fulllength_path="",
        medium_path="",
        small_path="",
        tags=[],
        url="",
        hash="",
        query="",
    )
    portrait_record_new = await portraits_repository_for_tests.insert(portrait_record)
    assert portrait_record_new.id is not None
    portrait_record_get = await portraits_repository_for_tests.get(portrait_record_new.id)
    assert portrait_record_get == portrait_record_new
