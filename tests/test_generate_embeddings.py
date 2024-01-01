import pytest
from mock import Mock

from portrait_search.core.mongodb import PyObjectId
from portrait_search.dependencies import Container
from portrait_search.embeddings import EmbeddingRepository
from portrait_search.embeddings.entities import EmbeddingRecord
from portrait_search.generate_embeddings import generate_embeddings
from portrait_search.portraits import PortraitRepository
from portrait_search.portraits.entities import PortraitRecord


@pytest.fixture
def portraits_repository_mock() -> Mock:
    m = Mock(spec=PortraitRepository)
    m.get_many.return_value = []
    return m


@pytest.fixture
def embeddings_repository_mock() -> Mock:
    m = Mock(spec=EmbeddingRepository)
    m.get_by_type.return_value = []
    return m


@pytest.fixture(autouse=True)
def container(container: Container, portraits_repository_mock: Mock, embeddings_repository_mock: Mock) -> None:
    container.init_resources()
    container.portrait_repository.override(portraits_repository_mock)
    container.embedding_repository.override(embeddings_repository_mock)
    container.wire(modules=["portrait_search.generate_embeddings"])


async def test_generate_embeddings__empty_database(embeddings_repository_mock: Mock) -> None:
    # GIVEN no records in the database
    # WHEN generate_embeddings is called
    await generate_embeddings()
    # THEN no records are inserted into embeddings_repository
    embeddings_repository_mock.insert_many.assert_called_once_with([])


async def test_generate_embeddings__no_embeddings(
    embeddings_repository_mock: Mock, portraits_repository_mock: Mock
) -> None:
    # GIVEN no records in the embeddings collection and one record in the portraits collection
    portrait_mock = Mock(id=PyObjectId(), description="some description", spec=PortraitRecord)
    portraits_repository_mock.get_many.return_value = [portrait_mock]
    # WHEN generate_embeddings is called
    await generate_embeddings()
    # THEN one records is inserted into embeddings_repository
    inserted_embeddings: list[EmbeddingRecord] = embeddings_repository_mock.insert_many.call_args[0][0]
    assert len(inserted_embeddings) == 1
    assert inserted_embeddings[0].portrait_id == portrait_mock.id
    assert inserted_embeddings[0].embedded_text == portrait_mock.description
    assert len(inserted_embeddings[0].embedding) == 768


async def test_generate_embeddings__new_and_old(
    embeddings_repository_mock: Mock, portraits_repository_mock: Mock
) -> None:
    # GIVEN one record in the embeddings collection and 2 record in the portraits collection
    portrait_mock1 = Mock(id=PyObjectId(), description="some description", spec=PortraitRecord)
    portrait_mock2 = Mock(id=PyObjectId(), description="some other description", spec=PortraitRecord)
    portraits_repository_mock.get_many.return_value = [portrait_mock1, portrait_mock2]

    embedding_mock = Mock(
        portrait_id=portrait_mock1.id,
        embedded_texts=[portrait_mock1.description],
        embeddings=[1.0],
        spec=EmbeddingRecord,
    )
    embeddings_repository_mock.get_by_type.return_value = [embedding_mock]
    # WHEN generate_embeddings is called
    await generate_embeddings()
    # THEN one records is inserted into embeddings_repository
    inserted_embeddings: list[EmbeddingRecord] = embeddings_repository_mock.insert_many.call_args[0][0]
    assert len(inserted_embeddings) == 1
    assert inserted_embeddings[0].portrait_id == portrait_mock2.id
    assert inserted_embeddings[0].embedded_text == portrait_mock2.description
    assert len(inserted_embeddings[0].embedding) == 768
