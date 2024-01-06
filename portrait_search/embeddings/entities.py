from pydantic import BaseModel

from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.core.mongodb import MongoDBRecord, PyObjectId


class EmbeddingRecord(BaseModel):
    portrait_id: PyObjectId

    embedding: list[float]
    embedded_text: str

    splitter_type: SplitterType
    embedder_type: EmbedderType

    experiment: str | None = None


class MongoEmbeddingRecord(EmbeddingRecord, MongoDBRecord):
    pass


class EmbeddingSimilarity(BaseModel):
    portrait_id: PyObjectId

    embedding: list[float]
    embedded_text: str
    query: list[float] | None = None
    query_text: str | None = None

    similarity: float

    def to_explanation(self) -> str:
        return f"Query: {self.query_text}\nPortrait text: {self.embedded_text}\nSimilarity: {self.similarity}"
