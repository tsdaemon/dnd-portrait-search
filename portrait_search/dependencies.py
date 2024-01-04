from dependency_injector import containers, providers

from portrait_search.core import Config, init_logging
from portrait_search.core.mongodb import get_connection, get_database
from portrait_search.data_sources import data_sources_from_yaml
from portrait_search.embeddings import EMBEDDERS, SPLITTERS, EmbeddingRepository
from portrait_search.open_ai import OpenAIClient
from portrait_search.portraits import PortraitRepository
from portrait_search.retrieval import AtlasRetriever


class Container(containers.DeclarativeContainer):
    config = providers.Object(Config())

    logging = providers.Resource(init_logging)

    mongodb_connection = providers.Resource(
        get_connection,
        url=config.provided.mongodb_uri,
    )

    db = providers.Resource(
        get_database,
        connection=mongodb_connection,
        database_name=config.provided.mongodb_database_name,
    )

    data_sources = providers.Resource(
        data_sources_from_yaml,
        config=config,
    )

    portrait_repository = providers.Factory(
        PortraitRepository,
        db=db,
    )

    embedding_repository = providers.Factory(
        EmbeddingRepository,
        db=db,
    )

    openai_client = providers.Factory(
        OpenAIClient,
        api_key=config.provided.openai_api_key,
    )

    splitter = providers.Resource(
        lambda config: SPLITTERS[config.splitter_type](),
        config=config,
    )
    embedder = providers.Resource(
        lambda config: EMBEDDERS[config.embedder_type](),
        config=config,
    )

    retriever = providers.Factory(
        AtlasRetriever,
        portrait_repository=portrait_repository,
        embedding_repository=embedding_repository,
        splitter=splitter,
        embedder=embedder,
    )
