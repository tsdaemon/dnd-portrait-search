from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dependency_injector import containers, providers

from portrait_search.data_source import data_sources_from_yaml
from portrait_search.portrait import PortraitService

from .core.config import Config


def get_db(url: str, database_name: str) -> AsyncIOMotorDatabase:
    return AsyncIOMotorClient(url)[database_name]


class Container(containers.DeclarativeContainer):
    config = Config()

    data_sources = providers.Resource(
        data_sources_from_yaml,
        config=config,
    )

    db = providers.Resource(
        get_db,
        url=config.mongodb_uri,
        database_name=config.mongodb_database_name,
    )

    portrait_service = providers.Factory(
        PortraitService,
        db=db,
    )
