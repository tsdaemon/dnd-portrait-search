import asyncio

from dependency_injector.wiring import Provide, inject

from portrait_search.core.config import Config
from portrait_search.dependencies import Container
from portrait_search.embeddings import Embedder, EmbeddingRepository, TextSplitter
from portrait_search.embeddings.t2v import portraits2embeddings
from portrait_search.portraits import PortraitRepository


@inject
async def generate_embeddings(
    portrait_repository: PortraitRepository = Provide[Container.portrait_repository],
    embedding_repository: EmbeddingRepository = Provide[Container.embedding_repository],
    splitter: TextSplitter = Provide[Container.splitter],
    embedder: Embedder = Provide[Container.embedder],
    config: Config = Provide[Container.config],
) -> None:
    # find existing embeddings
    existing_embeddings = await embedding_repository.get_by_type(
        embedder_type=config.embedder_type, splitter_type=config.splitter_type
    )
    existing_portraits = {embedding.portrait_id for embedding in existing_embeddings}
    all_portraits = await portrait_repository.get_many()
    portraits_without_embeddings_ids = {p.id for p in all_portraits} - existing_portraits
    portraits_without_embeddings = [p for p in all_portraits if p.id in portraits_without_embeddings_ids]
    embeddings = portraits2embeddings(
        portraits_without_embeddings, splitter, config.splitter_type, embedder, config.embedder_type
    )
    await embedding_repository.insert_many(embeddings)


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(generate_embeddings())
