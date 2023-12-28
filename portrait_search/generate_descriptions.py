import asyncio
from pathlib import Path

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from dependency_injector.wiring import Provide, inject

from portrait_search.data_sources import BaseDataSource
from portrait_search.dependencies import Container
from portrait_search.open_ai import OpenAIClient
from portrait_search.open_ai import PORTRAIT_DESCRIPTION_QUERY_V1
from portrait_search.portraits import PortraitRepository
from portrait_search.portraits import Portrait

# each worker uses ~10k tokens per minute, with a current limit 20k tokens it should be fine to use 2 workers
N_JOBS = 2
openai_semaphore = asyncio.Semaphore(N_JOBS)


async def _generate_portraint_description_and_store(
    portrait: Portrait,
    openai_client: OpenAIClient,
    portrait_repository: PortraitRepository,
):
    async with openai_semaphore:
        description = await openai_client.make_image_query(
            query=PORTRAIT_DESCRIPTION_QUERY_V1,
            image_path=portrait.fulllength_path,
        )

    portrait_record = portrait.to_record()
    portrait_record.description = description
    portrait_record.query = PORTRAIT_DESCRIPTION_QUERY_V1

    await portrait_repository.create(portrait_record)


@inject
async def generate_descriptions(
    local_data_folder: Path = Provide[Container.config.local_data_folder],
    data_sources: list[BaseDataSource] = Provide[Container.data_sources],
    portrait_repository: PortraitRepository = Provide[Container.portrait_repository],
    openai_client: OpenAIClient = Provide[Container.openai_client],
):
    if isinstance(local_data_folder, Provide):
        local_data_folder = local_data_folder.provider  # type: ignore

    print("Extracting portraits from sources...")
    portraits = []
    local_portraits_folder = local_data_folder / "portraits"
    for data_source in data_sources:
        portraits.extend(await data_source.retrieve(local_portraits_folder))
    print(f"Extracted {len(portraits)} portraits from data sources.")

    # calculate hashes of local images
    local_hashes_to_portraits = {}
    for portrait in tqdm(portraits, desc="Calculating portrait hashes"):
        # if hash already exists, we have a duplicate, and we just ignore it
        local_hashes_to_portraits[portrait.get_fulllength_hash()] = portrait

    print(f"Found {len(local_hashes_to_portraits)} unique portraits.")
    local_hashes = set(local_hashes_to_portraits.keys())
    existing_hashes = await portrait_repository.get_distinct_hashes()
    new_hashes = local_hashes - existing_hashes
    new_portraits = [local_hashes_to_portraits[hash_] for hash_ in new_hashes]
    print(f"Found {len(new_portraits)} new portraits.")

    # generate descriptions for new portraits in parallel
    tasks = [
        asyncio.create_task(
            _generate_portraint_description_and_store(
                portrait,
                openai_client,
                portrait_repository,
            )
        )
        for portrait in new_portraits
    ]
    await tqdm_asyncio.gather(*tasks, desc="Generating descriptions")

    print("Done!")


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(generate_descriptions())
