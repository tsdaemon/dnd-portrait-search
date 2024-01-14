import pytest

from portrait_search.quality.dataset import DatasetEntry, PortraitMatch, Query
from portrait_search.quality.judge import prepare_expected_results


@pytest.fixture
def dataset_entries() -> list[DatasetEntry]:
    return [
        DatasetEntry(
            name="test",
            queries=[
                Query(
                    query="test1",
                    portraits=[
                        PortraitMatch(path="path1", match={"match1"}),
                        PortraitMatch(path="path2", match={"match2"}),
                        PortraitMatch(path="path3", match={"match3"}),
                    ],
                    match={"match1", "match2", "match3"},
                ),
                Query(
                    query="test2",
                    portraits=[
                        PortraitMatch(path="path1", match={"match2", "match3"}),
                        PortraitMatch(path="path2", match={"match2"}),
                        PortraitMatch(path="path3", match={"match4"}),
                    ],
                    match={"match2", "match3", "match4"},
                ),
            ],
        )
    ]


def test_prepare_expected_results(dataset_entries: list[DatasetEntry]) -> None:
    expected_results = prepare_expected_results(dataset_entries)
    assert expected_results == {
        "test1": [
            ("path1", pytest.approx(0.45186276187)),
            ("path2", pytest.approx(0.27406861906)),
            ("path3", pytest.approx(0.27406861906)),
        ],
        "test2": [
            ("path1", pytest.approx(0.5481372)),
            ("path2", pytest.approx(0.2740686)),
            ("path3", pytest.approx(0.4518627)),
        ],
    }
