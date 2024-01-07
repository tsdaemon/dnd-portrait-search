from collections.abc import AsyncGenerator

import pytest

from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.core.mongodb import PyObjectId
from portrait_search.dependencies import Container
from portrait_search.embeddings.entities import MongoEmbeddingRecord
from portrait_search.embeddings.repository import MongoEmbeddingRepository


class TestMongoEmbeddingRepository:
    pytestmark = pytest.mark.usefixtures("inject_mongodb_database_for_test")

    @pytest.fixture
    def embedding_repository(self, container: Container) -> MongoEmbeddingRepository:
        return MongoEmbeddingRepository(container.db())

    @pytest.fixture
    async def existing_embedding_records(
        self, embedding_repository: MongoEmbeddingRepository
    ) -> AsyncGenerator[list[MongoEmbeddingRecord], None]:
        records = [
            MongoEmbeddingRecord(
                portrait_id=PyObjectId(),
                embedding=[1, 2, 3],
                embedder_type=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
                splitter_type=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
                embedded_text="spaghetti",
            ),
            MongoEmbeddingRecord(
                portrait_id=PyObjectId(),
                embedding=[-1, -2, -3],
                embedder_type=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
                splitter_type=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
                embedded_text="not spaghetti",
            ),
        ]
        records = await embedding_repository.insert_many(records)
        yield records
        await embedding_repository.delete(records[0].id)
        await embedding_repository.delete(records[1].id)

    async def test_get_by_type(
        self, embedding_repository: MongoEmbeddingRepository, existing_embedding_records: list[MongoEmbeddingRecord]
    ) -> None:
        # GIVEN 2 test records added to embeddings collection
        # WHEN getting embeddings by their type
        embeddings = await embedding_repository.get_by_type(
            splitter_type=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
            embedder_type=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
        )
        # THEN embeddings are 2
        assert len(embeddings) == 2
        # THEN embeddings are the same as the test records
        assert embeddings[0] == existing_embedding_records[0]
        assert embeddings[1] == existing_embedding_records[1]

    @pytest.mark.skip(reason="No vector search index in test database")
    async def test_vector_search(
        self, embedding_repository: MongoEmbeddingRepository, existing_embedding_records: list[MongoEmbeddingRecord]
    ) -> None:
        # GIVEN 2 test records added to embeddings collection
        # WHEN doing vector search
        embeddings = await embedding_repository.get_by_type(
            splitter_type=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
            embedder_type=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
        )
        # THEN embeddings are 2
        assert len(embeddings) == 2
        # THEN embeddings are the same as the test records
        assert embeddings[0] == existing_embedding_records[0]
        assert embeddings[1] == existing_embedding_records[1]
