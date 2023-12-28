from portrait_search.core import Config
from portrait_search.data_sources import data_sources_from_yaml


def test_data_sources_from_yaml():
    config = Config()
    data_sources = data_sources_from_yaml(config)
    assert len(data_sources) == 1
    assert all(d.url for d in data_sources)
