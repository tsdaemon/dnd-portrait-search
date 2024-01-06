import abc
from pathlib import Path

from portrait_search.core.config import Config
from portrait_search.portraits.entities import Portrait


class DataSourceError(Exception):
    pass


class BaseDataSource:
    def __init__(self, config: Config, url: str) -> None:
        self.config = config
        self.url = url

    @abc.abstractmethod
    async def retrieve(self, folder: Path) -> list[Portrait]:
        raise NotImplementedError()
