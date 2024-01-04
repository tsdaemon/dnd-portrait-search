from pydantic import BaseModel

from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.core.mongodb import MongoDBRecord, PyObjectId


class EmbeddingRecord(MongoDBRecord):
    portrait_id: PyObjectId

    embedding: list[float]
    embedded_text: str

    splitter_type: SplitterType
    embedder_type: EmbedderType


class EmbeddingSimilarity(BaseModel):
    portrait_id: PyObjectId

    embedding: list[float]
    embedded_text: str
    query: list[float] | None
    query_text: str | None

    similarity: float
