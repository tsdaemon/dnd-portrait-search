import pytest

from portrait_search.quality.metrics import METRICS


@pytest.fixture
def query_results() -> list[tuple[str, list[str]]]:
    return [
        ("path1", ["path1", "path2", "path3"]),
        ("path2", ["path1", "path2", "path3"]),
        ("path3", ["path2", "path1"]),
    ]


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("precision@k", 2 / 3),
        ("mean_reciprocal_rank", 0.5),
    ],
)
def test_metrics(metric: str, expected: float, query_results: list[tuple[str, list[str]]]) -> None:
    assert METRICS[metric](query_results) == pytest.approx(expected)
