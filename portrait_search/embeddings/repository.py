import abc
from collections import defaultdict
from pathlib import Path

import chromadb
from motor.motor_asyncio import AsyncIOMotorDatabase

from portrait_search.core.enums import DistanceType, EmbedderType, SplitterType
from portrait_search.core.mongodb import MongoDBRepository, PyObjectId
from portrait_search.embeddings.distance import DISTANCE_TO_SIMILARITY

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
        distance_type: DistanceType,
        experiment: str | None = None,
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        raise NotImplementedError()

    @abc.abstractmethod
    async def insert_many(self, records: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
        raise NotImplementedError()


class ChromaEmbeddingRepository(EmbeddingRepository):
    # map internally to avoid hitting the 64 character limit
    EMBEDDER_TYPE_MAPPTING = {
        EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS: "instrLargePFChar",
        EmbedderType.ALL_MINI_LM_L6_V2: "allMLML6v2",
        EmbedderType.MS_MARCO_DISTILBERT_BASE_V4: "msmarcoDBBv4",
        EmbedderType.MS_MARCO_ROBERTA_BASE_ANCE_FIRSTP: "msmarcoRBAFp",
    }
    SPLITTER_TYPE_MAPPING = {
        SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_120_OVERLAP_60: "langchainRec120o60",
        SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_160_OVERLAP_40_SAME_QUERY: "langchainRec160o40",
    }
    DISTANCE_TYPE_MAPPING = {
        DistanceType.EUCLIDEAN: "l2",
        DistanceType.COSINE: "cosine",
        DistanceType.DOT_PRODUCT: "ip",
    }

    def __init__(self, databases_path: Path):
        self.client = chromadb.PersistentClient(path=str(databases_path / "chroma.db"))

    def _distance_type_to_space(self, distance_type: DistanceType) -> str:
        if distance_type in self.DISTANCE_TYPE_MAPPING:
            return self.DISTANCE_TYPE_MAPPING[distance_type]
        raise ValueError(f"distance_type {distance_type} is not supported in Chroma")

    def get_collection(
        self, splitter_type: SplitterType, embedder_type: EmbedderType, distance_type: DistanceType
    ) -> chromadb.Collection:
        stype = self.SPLITTER_TYPE_MAPPING[splitter_type]
        etype = self.EMBEDDER_TYPE_MAPPTING[embedder_type]
        dtype = self._distance_type_to_space(distance_type)
        name = f"e-{stype}-{etype}-{dtype}"
        return self.client.get_or_create_collection(
            name, metadata={"hnsw:space": self._distance_type_to_space(distance_type)}
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

    def _chroma_query_to_embedding_similarity(
        self, query_result: chromadb.QueryResult, distance_type: DistanceType
    ) -> list[EmbeddingSimilarity]:
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
            similarity = DISTANCE_TO_SIMILARITY[distance_type](distance)
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
        collection = self.get_collection(splitter_type, embedder_type, DistanceType.EUCLIDEAN)
        records = collection.get(include=["documents", "embeddings", "metadatas"])
        return self._chroma_get_to_embedding_record(records, splitter_type, embedder_type)

    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        distance_type: DistanceType,
        experiment: str | None = None,
        limit: int = 10,
    ) -> list[EmbeddingSimilarity]:
        # experiment is ignored
        _ = experiment

        collection = self.get_collection(splitter_type, embedder_type, distance_type)
        records = collection.query(
            query_embeddings=query_vector,
            n_results=limit,
            include=["documents", "embeddings", "metadatas", "distances"],
        )
        similarities = self._chroma_query_to_embedding_similarity(records, distance_type)
        for similarity in similarities:
            similarity.query = query_vector

        return similarities

    async def insert_many(self, records: list[EmbeddingRecord]) -> list[EmbeddingRecord]:
        records_by_type = defaultdict(list)
        for record in records:
            record.id = PyObjectId()
            records_by_type[(record.splitter_type, record.embedder_type)].append(record)

        for similarity_type in DistanceType:
            for (splitter_type, embedder_type), records in records_by_type.items():
                collection = self.get_collection(splitter_type, embedder_type, similarity_type)
                ids = [str(record.id) for record in records]
                documents = [record.embedded_text for record in records]
                embeddings = [record.embedding for record in records]
                metadatas = [{"portrait_id": str(record.portrait_id)} for record in records]

                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,  # type: ignore
                    metadatas=metadatas,  # type: ignore
                )
        return records


class MongoEmbeddingRepository(MongoDBRepository[EmbeddingRecord]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__(db, EmbeddingRecord)

    @property
    def collection(self) -> str:
        return "embeddings"

    async def get_by_type(self, splitter_type: SplitterType, embedder_type: EmbedderType) -> list[EmbeddingRecord]:
        return await self.get_many(
            splitter_type=str(splitter_type),
            embedder_type=str(embedder_type),
        )

    async def vector_search(
        self,
        query_vector: list[float],
        splitter_type: SplitterType,
        embedder_type: EmbedderType,
        distance_type: DistanceType,
        experiment: str | None = None,
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
                        "index": f"portrait-embeddings-search-{distance_type}",
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
