import pytest

from portrait_search.embeddings import EMBEDDERS, Embedder
from portrait_search.embeddings.embedders import EXPECTED_DIMENSIONALITY


@pytest.mark.parametrize("embedder", EMBEDDERS.values(), ids=EMBEDDERS.keys())
def test_embedders(embedder: type[Embedder]) -> None:
    texts = [
        "The character depicted in the image appears to be a female elf, given the pointed ears and slender build. She possesses",  # noqa: E501
        "given the pointed ears and slender build. She possesses a demeanor that suggests a neutral alignment, focused more on",  # noqa: E501
        "demeanor that suggests a neutral alignment, focused more on balance or personal goals than strict adherence to good or",  # noqa: E501
    ]
    embeddings = embedder().embed(texts)
    assert len(embeddings) == len(texts)
    assert all(len(embedding) == EXPECTED_DIMENSIONALITY for embedding in embeddings)
