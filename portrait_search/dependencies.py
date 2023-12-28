from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dependency_injector import containers, providers
from portrait_search.core import Config
from portrait_search.core import init_logging

from portrait_search.data_sources import data_sources_from_yaml
from portrait_search.open_ai import OpenAIClient
from portrait_search.portraits import PortraitRepository


def get_db(url: str, database_name: str) -> AsyncIOMotorDatabase:
    return AsyncIOMotorClient(url)[database_name]


class Container(containers.DeclarativeContainer):
    config = Config()

    logging = providers.Resource(init_logging)

    data_sources = providers.Resource(
        data_sources_from_yaml,
        config=config,
    )

    db = providers.Resource(
        get_db,
        url=config.mongodb_uri,
        database_name=config.mongodb_database_name,
    )

    portrait_repository = providers.Factory(
        PortraitRepository,
        db=db,
    )

    openai_client = providers.Factory(
        OpenAIClient,
        api_key=config.openai_api_key,
    )
