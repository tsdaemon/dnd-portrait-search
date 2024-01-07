import abc
from pathlib import Path

import chromadb
from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.core.enums import EmbedderType, SimilarityType, SplitterType
from portrait_search.core.mongodb import MongoDBRepository, PyObjectId

from .entities import EmbeddingRecord, EmbeddingSimilarity


class EmbeddingRepository(abc.ABC):
    @abc.abstractmethod
    async def get_by_type(self, splitter_type: SplitterType, embedder_type: EmbedderType) -> list[EmbeddingRecord]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        method: SimilarityType,
        experiment: str | None = None,
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def insert_many(self, records: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
        raise NotImplementedError()


class ChromaEmbeddingRepository(EmbeddingRepository):
    def __init__(self, databases_path: Path):
        self.client = chromadb.PersistentClient(path=str(databases_path / "chroma.db"))

    def _similarity_type_to_space(self, similarity_type: SimilarityType) -> str:
        if similarity_type == SimilarityType.EUCLIDEAN:
            return "l2"
        if similarity_type == SimilarityType.COSINE:
            return "cosine"
        if similarity_type == SimilarityType.DOT_PRODUCT:
            return "ip"

        raise ValueError(f"similarity_type {similarity_type} is not supported in Chroma")

    def get_collection(
        self, splitter_type: SplitterType, embedder_type: EmbedderType, similarity_type: SimilarityType
    ) -> chromadb.Collection:
        name = f"e-{splitter_type.value}-{embedder_type.value}-{similarity_type.value}"
        return self.client.get_or_create_collection(
            name, metadata={"hnsw:space": self._similarity_type_to_space(similarity_type)}
        )

    def _chroma_get_to_embedding_record(
        self, get_result: chromadb.GetResult, splitter_type: SplitterType, embedder_type: EmbedderType
    ) -> list[EmbeddingRecord]:
        if get_result is None:
            return []
        if get_result["metadatas"] is None:
            raise ValueError("get_result must have metadatas")
        if get_result["documents"] is None:
            raise ValueError("get_result must have documents")
        if get_result["embeddings"] is None:
            raise ValueError("get_result must have embeddings")

        results = []
        for id, metatada, document, embedding in zip(
            get_result["ids"], get_result["metadatas"], get_result["documents"], get_result["embeddings"]
        ):
            if not isinstance(metatada["portrait_id"], str):
                raise ValueError("metatada['portrait_id'] must be str")

            results.append(
                EmbeddingRecord(
                    id=PyObjectId(id),  # type: ignore
                    portrait_id=PyObjectId(metatada["portrait_id"]),
                    embedding=list(embedding),
                    embedded_text=document,
                    splitter_type=splitter_type,
                    embedder_type=embedder_type,
                )
            )
        return results

    def _chroma_query_to_embedding_similarity(self, query_result: chromadb.QueryResult) -> list[EmbeddingSimilarity]:
        if query_result is None:
            return []
        if query_result["metadatas"] is None:
            raise ValueError("query_result must have metadatas")
        if query_result["documents"] is None:
            raise ValueError("query_result must have documents")
        if query_result["embeddings"] is None:
            raise ValueError("query_result must have embeddings")
        if query_result["distances"] is None:
            raise ValueError("query_result must have distances")

        results = []
        for id, metatada, document, embedding, distance in zip(
            query_result["ids"][0],
            query_result["metadatas"][0],
            query_result["documents"][0],
            query_result["embeddings"][0],
            query_result["distances"][0],
        ):
            if not isinstance(metatada["portrait_id"], str):
                raise ValueError("metatada['portrait_id'] must be str")
            similarity = 1 / (1 + distance)
            results.append(
                EmbeddingSimilarity(
                    id=PyObjectId(id),  # type: ignore
                    portrait_id=PyObjectId(metatada["portrait_id"]),
                    embedding=list(embedding),
                    embedded_text=document,
                    similarity=similarity,
                )
            )
        return results

    async def get_by_type(self, splitter_type: SplitterType, embedder_type: EmbedderType) -> list[EmbeddingRecord]:
        # Similarity does not matter in this context, data should be replicated across all spaces
        collection = self.get_collection(splitter_type, embedder_type, SimilarityType.EUCLIDEAN)
        records = collection.get(include=["documents", "embeddings", "metadatas"])
        return self._chroma_get_to_embedding_record(records, splitter_type, embedder_type)

    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        method: SimilarityType,
        experiment: str | None = None,
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        # experiment is ignored
        _ = experiment

        collection = self.get_collection(splitter_type, embedder_type, method)
        records = collection.query(
            query_embeddings=query_vector,
            n_results=limit,
            include=["documents", "embeddings", "metadatas", "distances"],
        )
        similarities = self._chroma_query_to_embedding_similarity(records)
        for similarity in similarities:
            similarity.query = query_vector

        return similarities

    async def insert_many(self, records: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
        for record in records:
            record.id = PyObjectId()
            for similarity_type in SimilarityType:
                collection = self.get_collection(record.splitter_type, record.embedder_type, similarity_type)
                collection.add(
                    ids=[str(record.id)],
                    documents=[record.embedded_text],
                    embeddings=[record.embedding],  # type: ignore
                    metadatas=[{"portrait_id": str(record.portrait_id)}],
                )
        return records


class MongoEmbeddingRepository(MongoDBRepository[EmbeddingRecord]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, EmbeddingRecord)

    @property
    def collection(self) -> str:
        return "embeddings"

    async def get_by_type(self, splitter_type: SplitterType, embedder_type: EmbedderType) -> list[EmbeddingRecord]:
        # legacy long names
        if splitter_type == SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60:
            splitter_type_s = "langchain-recursive-text-splitter-chunk-120-overlap-60"
        else:
            splitter_type_s = str(splitter_type)

        if embedder_type == EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS:
            embedder_type_s = "instructor-large-pathfinder-character-instructions"
        else:
            embedder_type_s = str(embedder_type)

        return await self.get_many(
            splitter_type=splitter_type_s,
            embedder_type=embedder_type_s,
        )

    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        experiment: str | None = None,
        method: str = "euclidean",
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        """Returns a list of Embeddings and their similarities that match the query vector."""
        filter = [
            {"splitter_type": splitter_type},
            {"embedder_type": embedder_type},
        ]
        if experiment:
            filter.append({"experiment": experiment})
        entities = self.db[self.collection].aggregate(
            [
                {
                    "$vectorSearch": {
                        "index": f"portrait-embeddings-search-{method}",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": limit * 20,
                        "limit": limit,
                        "filter": {
                            "$and": filter  # for some reason, $and is required here
                        },
                    }
                },
                {
                    "$project": {
                        "portrait_id": 1,
                        "embedding": 1,
                        "embedded_text": 1,
                        "similarity": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
        )
        return [EmbeddingSimilarity.model_validate(entity) async for entity in entities]
