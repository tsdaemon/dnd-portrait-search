from itertools import product

import pytest
from bson import ObjectId

from portrait_search.core.config import EmbedderType, SplitterType
from portrait_search.embeddings import (
    EMBEDDERS,
    SPLITTERS,
    Embedder,
    TextSplitter,
    portraits2embeddings,
    query2embeddings,
)
from portrait_search.portraits import PortraitRecord


@pytest.mark.parametrize("embedder_type, splitter_type", product(EMBEDDERS, SPLITTERS))
def test_portraits2embeddings(
    embedder_type: EmbedderType, splitter_type: SplitterType, portrait_description_example: str
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
    splitter = SPLITTERS[splitter_type]()
    embedder = EMBEDDERS[embedder_type]()
    embeddings_records = portraits2embeddings(portraits_records, splitter, embedder)

    # THEN the result is a list of 38 EmbeddingRecords, first 19 with the first portrait id, last 19 with the second
    assert len(embeddings_records) == 38
    assert embeddings_records[0].portrait_id == portraits_records[0].id
    assert embeddings_records[19].portrait_id == portraits_records[1].id

    # THEN first 19 embeddings are equal to the last 19 embeddings
    assert embeddings_records[0].embedding == embeddings_records[19].embedding
    assert embeddings_records[0].embedded_text == embeddings_records[19].embedded_text

    # THEN length of all embedding vectors is 768
    assert all(len(er.embedding) == 768 for er in embeddings_records)

    # THEN splitter_type and embedder_type are set correctly
    assert all(e.splitter_type == splitter_type for e in embeddings_records)
    assert all(e.embedder_type == embedder_type for e in embeddings_records)


@pytest.mark.parametrize("embedder_type, splitter_type", product(EMBEDDERS.values(), SPLITTERS.values()))
def test_query2embeddings(embedder_type: type[Embedder], splitter_type: type[TextSplitter]) -> None:
    # GIVEN a query string
    query = "A rogue elf female with a knife"

    # WHEN query2embeddings is called
    embeddings, texts = query2embeddings(query, splitter_type(), embedder_type())

    # THEN all texts non-empty
    assert all(len(t) > 0 for t in texts)

    # THEN all texts are present in the query
    assert all(t in query for t in texts)

    # THEN the result is a list of vectors
    assert len(embeddings) > 1

    # THEN length of all embedding vectors is 768
    assert all(len(e) == 768 for e in embeddings)
