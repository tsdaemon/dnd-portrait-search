import pytest

from portrait_search.quality.metrics import METRICS, QUERY_RESULT_FOR_METRIC


@pytest.fixture
def query_results() -> list[QUERY_RESULT_FOR_METRIC]:
    return [
        (["path1", "path2", "path3"], [("path2", 1), ("path3", 1), ("path4", 1)]),
        (["path2", "path3", "path4"], [("path1", 1), ("path2", 1), ("path4", 1)]),
        (["path1", "path2", "path3"], [("path2", 0.1), ("path3", 0.8), ("path4", 0.2)]),
        (["path10", "path20", "path30"], [("path2", 0.1), ("path3", 0.8), ("path4", 0.2)]),
    ]


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("precision@k", (2 / 3 + 2 / 3) / 4),
        ("mrr", (1 / 2 + 1) / 4),
        ("w_rel_precision@k", (2 / 3 + 2 / 3 + (0.8 + 0.1) / 3) / 4),
        ("w_rel_mrr", (1 / 2 + 1 + 0.1 / 2) / 4),
    ],
)
def test_metrics(metric: str, expected: float, query_results: list[QUERY_RESULT_FOR_METRIC]) -> None:
    assert METRICS[metric](query_results) == pytest.approx(expected)
