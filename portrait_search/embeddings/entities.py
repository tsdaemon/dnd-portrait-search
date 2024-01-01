from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.core.mongodb import MongoDBRecord, PyObjectId


class EmbeddingRecord(MongoDBRecord):
    portrait_id: PyObjectId

    embedding: list[float]
    embedded_text: str

    splitter_type: SplitterType
    embedder_type: EmbedderType
