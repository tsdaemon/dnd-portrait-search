import asyncio
from itertools import product

from dependency_injector.wiring import Provide, inject

from portrait_search.dependencies import Container
from portrait_search.embeddings.embedders import EMBEDDERS
from portrait_search.embeddings.repository import EmbeddingRepository
from portrait_search.embeddings.splitters import SPLITTERS
from portrait_search.embeddings.t2v import portraits2embeddings
from portrait_search.portraits.repository import PortraitRepository


@inject
async def generate_embeddings(
    portrait_repository: PortraitRepository = Provide[Container.portrait_repository],
    embedding_repository: EmbeddingRepository = Provide[Container.embedding_repository],
    experiment: str = Provide[Container.config.provided.experiment],
) -> None:
    all_splitters_and_providers = product(SPLITTERS.values(), EMBEDDERS.values())
    all_portraits = await portrait_repository.get_many()
    for splitter, embedder in all_splitters_and_providers:
        print(f"Generating embeddings for splitter {splitter.type}, embedder {embedder.type}")
        # find existing embeddings
        existing_embeddings = await embedding_repository.get_by_type(
            embedder_type=embedder.type, splitter_type=splitter.type
        )
        print("Already generated: ", len(existing_embeddings))
        existing_portraits = {embedding.portrait_id for embedding in existing_embeddings}

        portraits_without_embeddings_ids = {p.id for p in all_portraits} - existing_portraits
        portraits_without_embeddings = [p for p in all_portraits if p.id in portraits_without_embeddings_ids]
        embeddings = portraits2embeddings(portraits_without_embeddings, splitter, embedder)
        print("Generated: ", len(embeddings))
        if experiment:
            for embedding in embeddings:
                embedding.experiment = experiment
        await embedding_repository.insert_many(embeddings)
        print("Done!")
        print("-----")


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(generate_embeddings())
