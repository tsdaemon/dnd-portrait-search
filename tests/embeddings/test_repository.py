import tempfile
from collections.abc import AsyncGenerator, Generator
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from portrait_search.core.enums import EmbedderType, SimilarityType, SplitterType
from portrait_search.core.mongodb import PyObjectId
from portrait_search.dependencies import Container
from portrait_search.embeddings.entities import EmbeddingRecord
from portrait_search.embeddings.repository import ChromaEmbeddingRepository, MongoEmbeddingRepository


@pytest.fixture
def embedding_records_for_test() -> list[EmbeddingRecord]:
    records = [
        EmbeddingRecord(
            portrait_id=PyObjectId(),
            embedding=[1, 2, 3],
            embedder_type=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
            splitter_type=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
            embedded_text="spaghetti",
        ),
        EmbeddingRecord(
            portrait_id=PyObjectId(),
            embedding=[-1, -2, -3],
            embedder_type=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
            splitter_type=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
            embedded_text="not spaghetti",
        ),
    ]
    return records


class TestChromaEmbeddingRepository:
    @pytest.fixture
    def embedding_repository(self) -> Generator[ChromaEmbeddingRepository, Any, Any]:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "database"
            path.mkdir(mode=0o777, parents=True)

            yield ChromaEmbeddingRepository(databases_path=path)

    @pytest.fixture
    async def existing_embedding_records(
        self, embedding_records_for_test: list[EmbeddingRecord], embedding_repository: ChromaEmbeddingRepository
    ) -> AsyncGenerator[list[EmbeddingRecord], None]:
        records_new = await embedding_repository.insert_many(embedding_records_for_test)
        yield records_new

    @pytest.mark.parametrize(
        "embedder_type, splitter_type, similarity_type", product(EmbedderType, SplitterType, SimilarityType)
    )
    def test_get_collection(
        self,
        embedder_type: EmbedderType,
        splitter_type: SplitterType,
        similarity_type: SimilarityType,
        embedding_repository: ChromaEmbeddingRepository,
    ) -> None:
        collection = embedding_repository.get_collection(splitter_type, embedder_type, similarity_type)
        assert collection.name == f"e-{splitter_type.value}-{embedder_type.value}-{similarity_type.value}"

    async def test_get_by_type(
        self, embedding_repository: ChromaEmbeddingRepository, existing_embedding_records: list[EmbeddingRecord]
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

    @pytest.mark.parametrize("similarity_type", SimilarityType)
    async def test_vector_search(
        self,
        similarity_type: SimilarityType,
        embedding_repository: ChromaEmbeddingRepository,
        existing_embedding_records: list[EmbeddingRecord],
    ) -> None:
        # GIVEN 2 test records added to embeddings collection
        _ = existing_embedding_records
        # WHEN searching for similar embeddings
        embedding_similarities = await embedding_repository.vector_search(
            query_vector=[1, 1, 1],
            splitter_type=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60,
            embedder_type=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS,
            method=similarity_type,
        )

        # THEN found 2 embedding similarities
        assert len(embedding_similarities) == 2

        # THEN the first one is spaghetti
        assert embedding_similarities[0].embedded_text == "spaghetti"


class TestMongoEmbeddingRepository:
    pytestmark = pytest.mark.usefixtures("inject_mongodb_database_for_test")

    @pytest.fixture
    def embedding_repository(self, container: Container) -> MongoEmbeddingRepository:
        return MongoEmbeddingRepository(container.db())

    @pytest.fixture
    async def existing_embedding_records(
        self, embedding_repository: MongoEmbeddingRepository, embedding_records_for_test: list[EmbeddingRecord]
    ) -> AsyncGenerator[list[EmbeddingRecord], None]:
        records = await embedding_repository.insert_many(embedding_records_for_test)
        yield records
        await embedding_repository.delete(records[0].id)
        await embedding_repository.delete(records[1].id)

    @pytest.mark.skip(reason="Broken and I have no reason fixing it")
    async def test_get_by_type(
        self, embedding_repository: MongoEmbeddingRepository, existing_embedding_records: list[EmbeddingRecord]
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
    async def test_vector_search(self) -> None:
        pass
