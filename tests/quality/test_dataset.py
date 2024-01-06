from portrait_search.quality.dataset import load_dataset


def test_load_dataset() -> None:
    # GIVEN a dataset in folder v1
    # WHEN loading the dataset
    dataset = load_dataset("v1")

    # THEN the dataset is not empty
    assert len(dataset) > 0

    # THEN all entries have queries
    assert all(len(entry.queries) > 0 for entry in dataset)

    # THEN all entries have tags
    assert all(len(entry.tags) > 0 for entry in dataset)

    # THEN all entries have a path
    assert all(entry.path != "" for entry in dataset)
