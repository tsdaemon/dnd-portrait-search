from pydantic import BaseModel, ConfigDict, Field
from pydantic_mongo import ObjectIdField  # type: ignore


class EmbeddingRecord(BaseModel):
    id: ObjectIdField | None = Field(alias="_id", default=None)
    portrait_id: ObjectIdField

    embeddings: list[list[float]]
    embedded_texts: list[str]

    splitter_class: str
    embedding_model_class: str

    model_config = ConfigDict(populate_by_name=True)
