import pytest

from portrait_search.embeddings import Embedder, EMBEDDERS


@pytest.mark.parametrize("embedder", EMBEDDERS.values(), ids=EMBEDDERS.keys())
def test_embedders(embedder: type[Embedder]) -> None:
    texts = [
        "The character depicted in the image appears to be a female elf, given the pointed ears and slender build. She possesses",
        "given the pointed ears and slender build. She possesses a demeanor that suggests a neutral alignment, focused more on",
        "demeanor that suggests a neutral alignment, focused more on balance or personal goals than strict adherence to good or",
    ]
    embeddings = embedder().embed(texts)
    assert len(embeddings) == len(texts)
    assert all(len(embedding) == 768 for embedding in embeddings)