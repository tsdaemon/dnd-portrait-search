import asyncio

from dependency_injector.wiring import Provide, inject

from portrait_search.dependencies import Container
from portrait_search.embeddings.embedders import Embedder
from portrait_search.embeddings.repository import EmbeddingRepository
from portrait_search.embeddings.splitters import Splitter
from portrait_search.embeddings.t2v import portraits2embeddings
from portrait_search.portraits.repository import PortraitRepository


@inject
async def generate_embeddings(
    portrait_repository: PortraitRepository = Provide[Container.portrait_repository],
    embedding_repository: EmbeddingRepository = Provide[Container.embedding_repository],
    splitter: Splitter = Provide[Container.splitter],
    embedder: Embedder = Provide[Container.embedder],
) -> None:
    # find existing embeddings
    existing_embeddings = await embedding_repository.get_by_type(
        embedder_type=embedder.embedder_type(), splitter_type=splitter.splitter_type()
    )
    existing_portraits = {embedding.portrait_id for embedding in existing_embeddings}
    all_portraits = await portrait_repository.get_many()
    portraits_without_embeddings_ids = {p.id for p in all_portraits} - existing_portraits
    portraits_without_embeddings = [p for p in all_portraits if p.id in portraits_without_embeddings_ids]
    embeddings = portraits2embeddings(portraits_without_embeddings, splitter, embedder)
    await embedding_repository.insert_many(embeddings)


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(generate_embeddings())
