import pytest

from portrait_search.embeddings import TextSplitter, SPLITTERS


@pytest.mark.parametrize("splitter", SPLITTERS.values(), ids=SPLITTERS.keys())
def test_splitters(
    splitter: type[TextSplitter], portrait_description_example: str
) -> None:
    texts = splitter().split(portrait_description_example)
    assert len(texts) > 1
