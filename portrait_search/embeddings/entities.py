from pydantic_mongo import ObjectIdField

from portrait_search.core.mongodb import MongoDBRecord


class EmbeddingRecord(MongoDBRecord):
    portrait_id: ObjectIdField

    embeddings: list[list[float]]
    embedded_texts: list[str]

    splitter_class: str
    embedding_model_class: str
