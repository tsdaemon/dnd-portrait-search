import pytest

from portrait_search.dependencies import Container
from portrait_search.portraits.entities import PortraitRecord
from portrait_search.portraits.repository import PortraitRepository

pytestmark = pytest.mark.usefixtures("inject_mongodb_database_for_test")


@pytest.fixture
def portraits_repository(container: Container) -> PortraitRepository:
    return container.portrait_repository()


async def test_get_hashes(portraits_repository: PortraitRepository) -> None:
    # GIVEN no records in the collection
    # WHEN get_distinct_hashes is called
    hashes = await portraits_repository.get_distinct_hashes()
    # THEN hashes is an empty set
    assert len(hashes) == 0

    # GIVEN two records in the collection
    portrait1 = PortraitRecord(
        fulllength_path="fulllength_path",
        medium_path="medium_path",
        small_path="small_path",
        tags=["tag1"],
        url="url1",
        hash="hash1",
        query="query1",
        description="description1",
    )
    portrait2 = PortraitRecord(
        fulllength_path="fulllength_path",
        medium_path="medium_path",
        small_path="small_path",
        tags=["tag2"],
        url="url2",
        hash="hash2",
        query="query2",
        description="description2",
    )
    await portraits_repository.insert_one(portrait1)
    await portraits_repository.insert_one(portrait2)

    # WHEN get_distinct_hashes is called
    hashes = await portraits_repository.get_distinct_hashes()
    # THEN hashes contains the hashes of the two records
    assert len(hashes) == 2
    assert "hash1" in hashes
    assert "hash2" in hashes
