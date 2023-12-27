import abc
from pathlib import Path
from portrait_search.core import Config
from portrait_search.portrait.entity import Portrait


class DataSourceError(Exception):
    pass


class BaseDataSource:
    def __init__(self, config: Config) -> None:
        self.config = config

    @abc.abstractmethod
    async def retrieve(self, folder: Path) -> list[Portrait]:
        raise NotImplementedError()
