from itertools import product
from bson import ObjectId
import pytest

from portrait_search.embeddings import EMBEDDERS, SPLITTERS, texts2vectors
from portrait_search.portraits import PortraitRecord


@pytest.mark.parametrize("embedder_name, splitter_name", product(EMBEDDERS, SPLITTERS))
def test_texts2vectors(
    embedder_name: str, splitter_name: str, portrait_description_example: str
) -> None:
    # GIVEN a list of PortraitRecords with 2 identical records
    portraits_records = [
        PortraitRecord(
            id=ObjectId(),  # type: ignore
            description=portrait_description_example,
            fulllength_path="",
            medium_path="",
            small_path="",
            tags=[],
            url="",
            hash="",
            query="",
        ),
        PortraitRecord(
            id=ObjectId(),  # type: ignore
            description=portrait_description_example,
            fulllength_path="",
            medium_path="",
            small_path="",
            tags=[],
            url="",
            hash="",
            query="",
        ),
    ]

    # WHEN texts2vectors is called with the portraits and the splitter and embedder
    embeddings_records = texts2vectors(portraits_records, splitter_name, embedder_name)

    # THEN the result is a list of 2 EmbeddingRecords with the same embeddings and embedded_texts
    assert len(embeddings_records) == 2
    assert embeddings_records[0].embeddings == embeddings_records[1].embeddings
    assert embeddings_records[0].embedded_texts == embeddings_records[1].embedded_texts

    # THEN embedding records has corresponing portrait ids
    assert embeddings_records[0].portrait_id == portraits_records[0].id
    assert embeddings_records[1].portrait_id == portraits_records[1].id

    # THEN embedding records has the correct splitter and embedder classes
    assert embeddings_records[0].splitter_class == splitter_name
    assert embeddings_records[0].embedding_model_class == embedder_name
    assert embeddings_records[1].splitter_class == splitter_name
    assert embeddings_records[1].embedding_model_class == embedder_name
