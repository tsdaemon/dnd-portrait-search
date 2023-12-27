import abc

from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, Field, TypeAdapter
import yaml

from portrait_search.core import Config

from .base import BaseDataSource
from .nexus import NexusDataSource


class BaseDataSourceConfig(BaseModel):
    kind: str

    @abc.abstractmethod
    def get_data_source(self, config: Config) -> BaseDataSource:
        raise NotImplementedError()


class NexusDataSourceConfig(BaseDataSourceConfig):
    kind: Literal["nexusmods"]
    game: str
    mod: int
    file: int

    def get_data_source(self, config: Config) -> BaseDataSource:
        return NexusDataSource(config, self.game, self.mod, self.file)


DataSourceConfigType = Annotated[NexusDataSourceConfig, Field(discriminator="kind")]


def data_sources_from_yaml(config: Config) -> list[BaseDataSource]:
    data_sources_config_path = Path(__file__).parent / "data_sources.yaml"

    with open(data_sources_config_path, "r") as file:
        data_sources_data = yaml.safe_load(file)

    data_sources: list[BaseDataSource] = []
    for data_source_config_data in data_sources_data["sources"]:
        adapter = TypeAdapter(DataSourceConfigType)
        data_source_config = adapter.validate_python(data_source_config_data)
        data_source = data_source_config.get_data_source(config)
        data_sources.append(data_source)

    return data_sources
