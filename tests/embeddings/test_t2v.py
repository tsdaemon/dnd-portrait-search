from itertools import product

import pytest
from bson import ObjectId

from portrait_search.core.enums import EmbedderType, SplitterType
from portrait_search.embeddings.embedders import EMBEDDERS, Embedder
from portrait_search.embeddings.splitters import SPLITTERS, Splitter
from portrait_search.embeddings.t2v import portraits2embeddings, query2embeddings
from portrait_search.portraits.entities import PortraitRecord


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
    splitter = SPLITTERS[splitter_type]
    embedder = EMBEDDERS[embedder_type]
    embeddings_records = portraits2embeddings(portraits_records, splitter, embedder)

    # THEN the result is a list of EmbeddingRecords, first half with the first portrait id, last half with the second
    assert len(embeddings_records) == 2 * len(splitter.split(portrait_description_example))
    for i in range(len(embeddings_records)):
        assert embeddings_records[i].portrait_id == portraits_records[i // int(len(embeddings_records) / 2)].id

    # THEN first half of embeddings are equal to the last half of embeddings
    assert all(
        embeddings_records[i].embedding == embeddings_records[i + int(len(embeddings_records) / 2)].embedding
        for i in range(int(len(embeddings_records) / 2))
    )

    # THEN all embedding vectors are non-empty
    assert all(len(er.embedding) > 0 for er in embeddings_records)

    # THEN splitter_type and embedder_type are set correctly
    assert all(e.splitter_type == splitter_type for e in embeddings_records)
    assert all(e.embedder_type == embedder_type for e in embeddings_records)


@pytest.mark.parametrize("embedder, splitter", product(EMBEDDERS.values(), SPLITTERS.values()))
def test_query2embeddings(embedder: Embedder, splitter: Splitter) -> None:
    # GIVEN a query string
    query = "A rogue elf female with a knife"

    # WHEN query2embeddings is called
    embeddings, texts = query2embeddings(query, splitter, embedder)

    # THEN all texts non-empty
    assert all(len(t) > 0 for t in texts)

    # THEN all texts are present in the query
    assert all(t in query for t in texts)

    # THEN the result is a list of vectors
    assert len(embeddings) > 0

    # THEN all embedding vectors are non-empty
    assert all(len(e) > 0 for e in embeddings)
