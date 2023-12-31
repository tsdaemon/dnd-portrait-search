from dependency_injector import containers, providers

from portrait_search.core import Config, init_logging
from portrait_search.core.mongodb import get_connection, get_database
from portrait_search.data_sources import data_sources_from_yaml
from portrait_search.open_ai import OpenAIClient
from portrait_search.portraits import PortraitRepository


class Container(containers.DeclarativeContainer):
    config = Config()

    logging = providers.Resource(init_logging)

    mongodb_connection = providers.Resource(
        get_connection,
        url=config.mongodb_uri,
    )

    db = providers.Resource(
        get_database,
        connection=mongodb_connection,
        database_name=config.mongodb_database_name,
    )

    data_sources = providers.Resource(
        data_sources_from_yaml,
        config=config,
    )

    portrait_repository = providers.Factory(
        PortraitRepository,
        db=db,
    )

    openai_client = providers.Factory(
        OpenAIClient,
        api_key=config.openai_api_key,
    )
