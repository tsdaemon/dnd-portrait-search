from unittest.mock import Mock, call

import pytest

from portrait_search.core.enums import DistanceType
from portrait_search.core.mongodb import PyObjectId
from portrait_search.embeddings.embedders import Embedder
from portrait_search.embeddings.entities import EmbeddingSimilarity
from portrait_search.embeddings.repository import EmbeddingRepository
from portrait_search.embeddings.splitters import Splitter
from portrait_search.portraits.repository import PortraitRepository
from portrait_search.retrieval.similarity import SimilarityRetriever


@pytest.fixture
def portraits_repository_mock() -> Mock:
    m = Mock(spec=PortraitRepository)
    m.get_one.return_value = None
    return m


@pytest.fixture
def embedding_repository_mock() -> Mock:
    m = Mock(spec=EmbeddingRepository)
    m.vector_search.return_value = []
    return m


@pytest.fixture
def splitter_mock() -> Mock:
    m = Mock(spec=Splitter)
    m.split_query.return_value = []
    return m


@pytest.fixture
def embedder_mock() -> Mock:
    m = Mock(spec=Embedder)
    m.embed.return_value = []
    return m


@pytest.fixture
def similarity_retriever(
    portraits_repository_mock: Mock,
    embedding_repository_mock: Mock,
    splitter_mock: Mock,
    embedder_mock: Mock,
) -> SimilarityRetriever:
    return SimilarityRetriever(
        distance_type=DistanceType.EUCLIDEAN,
        portrait_repository=portraits_repository_mock,
        embedding_repository=embedding_repository_mock,
        splitter=splitter_mock,
        embedder=embedder_mock,
    )


async def test_get_portraits__no_results(similarity_retriever: SimilarityRetriever) -> None:
    assert await similarity_retriever.get_portraits("test") == ([], [])


async def test_get_portraits__several_results(
    similarity_retriever: SimilarityRetriever,
    splitter_mock: Mock,
    embedder_mock: Mock,
    embedding_repository_mock: Mock,
    portraits_repository_mock: Mock,
) -> None:
    # GIVEN a query
    query = "A rogue elf female with a knife"
    # GIVEN splitter mock splits query into 2 parts with overlap
    splitter_mock.split_query.return_value = ["A rogue elf female", "female with a knife"]
    # GIVEN embedder mock returns 2 embeddings
    embedder_mock.embed.return_value = [[1, 2, 3], [4, 5, 6]]
    # GIVEN total number of unique portraits in the database is 10
    unique_portrait_ids = [PyObjectId() for _ in range(10)]
    # GIVEN embeddings repository returns 5 results for the first embedding and 3 for the second
    embedding_repository_mock.vector_search.side_effect = [
        [
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some original text 1",
                similarity=0.9,
                portrait_id=unique_portrait_ids[0],
            ),
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some original text 2",
                similarity=0.8,
                portrait_id=unique_portrait_ids[1],
            ),
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some original text 3",
                similarity=0.9,
                portrait_id=unique_portrait_ids[2],
            ),
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some original text 4",
                similarity=0.6,
                portrait_id=unique_portrait_ids[3],
            ),
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some original text 5",
                similarity=0.5,
                portrait_id=unique_portrait_ids[4],
            ),
        ],
        [
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some other original text ",
                similarity=0.5,
                portrait_id=unique_portrait_ids[8],
            ),
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some other original text 2",
                similarity=0.4,
                portrait_id=unique_portrait_ids[2],
            ),
            EmbeddingSimilarity(
                embedding=[],
                embedded_text="some other original text 3",
                similarity=0.3,
                portrait_id=unique_portrait_ids[1],
            ),
        ],
    ]
    # GIVEN portrait repository returns some string for each unique portrait id
    portraits_repository_mock.get_one.side_effect = lambda pid: f"Portrait {pid}"

    # WHEN get_portraits is called with the query and limit 2
    portraits, explanations = await similarity_retriever.get_portraits(query, limit=2)

    # THEN the splitter is called with the query
    splitter_mock.split_query.assert_called_once_with(query)
    # THEN the embedder is called with the 2 parts of the query
    embedder_mock.embed.assert_called_once_with(["A rogue elf female", "female with a knife"])
    # THEN the embeddings repository vector search is called twice with the 2 queries
    embedding_repository_mock.vector_search.assert_has_calls(
        [
            call(
                [1, 2, 3],
                splitter_mock.splitter_type(),
                embedder_mock.embedder_type(),
                DistanceType.EUCLIDEAN,
                experiment=None,
                limit=6,
            ),
            call(
                [4, 5, 6],
                splitter_mock.splitter_type(),
                embedder_mock.embedder_type(),
                DistanceType.EUCLIDEAN,
                experiment=None,
                limit=6,
            ),
        ]
    )
    # THEN the portrait repository is called 2 times with portrait id 0 and portrait id 3
    portraits_repository_mock.get_one.assert_has_calls(
        [
            call(unique_portrait_ids[0]),
            call(unique_portrait_ids[2]),
        ]
    )
    # THEN the portraits returned are the 2 portraits from the repository
    assert portraits == [f"Portrait {unique_portrait_ids[0]}", f"Portrait {unique_portrait_ids[2]}"]
    # THEN the explanations returned are the 2 explanations from the repository
    assert explanations == [
        [
            EmbeddingSimilarity(
                portrait_id=unique_portrait_ids[0],
                embedding=[],
                embedded_text="some original text 1",
                query=[1, 2, 3],
                query_text="A rogue elf female",
                similarity=0.9,
            )
        ],
        [
            EmbeddingSimilarity(
                portrait_id=unique_portrait_ids[2],
                embedding=[],
                embedded_text="some original text 3",
                query=[1, 2, 3],
                query_text="A rogue elf female",
                similarity=0.9,
            ),
            EmbeddingSimilarity(
                portrait_id=unique_portrait_ids[2],
                embedding=[],
                embedded_text="some other original text 2",
                query=[4, 5, 6],
                query_text="female with a knife",
                similarity=0.4,
            ),
        ],
    ]
