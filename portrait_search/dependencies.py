from dependency_injector import containers, providers

from portrait_search.core.config import Config
from portrait_search.core.logging import init_logging
from portrait_search.core.mongodb import get_connection, get_database
from portrait_search.data_sources.config import data_sources_from_yaml
from portrait_search.embeddings.embedders import EMBEDDERS
from portrait_search.embeddings.repository import EmbeddingRepository, MongoEmbeddingRepository
from portrait_search.embeddings.splitters import SPLITTERS
from portrait_search.open_ai.client import OpenAIClient
from portrait_search.portraits.repository import PortraitRepository
from portrait_search.retrieval.retriever import Retriever
from portrait_search.retrieval.similarity import SimilarityRetriever


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

    # Embedding repository
    # Known mypy issue: can't use abstract classes as dependencies
    # https://github.com/ets-labs/python-dependency-injector/issues/497
    embedding_repository = providers.Dependency(EmbeddingRepository)  # type: ignore[type-abstract]
    mongo_embedding_repository = providers.Factory(
        MongoEmbeddingRepository,
        db=db,
    )
    embedding_repository.override(mongo_embedding_repository)

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

    # Retriever
    retriever = providers.Dependency(Retriever)  # type: ignore[type-abstract]
    vector_similarity_retriever = providers.Factory(
        SimilarityRetriever,
        portrait_repository=portrait_repository,
        embedding_repository=embedding_repository,
        splitter=splitter,
        embedder=embedder,
    )

    retriever.override(vector_similarity_retriever)
