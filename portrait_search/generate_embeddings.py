import asyncio

from dependency_injector.wiring import inject

from portrait_search.dependencies import Container


@inject
async def generate_embeddings() -> None:
    pass


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(generate_embeddings())
