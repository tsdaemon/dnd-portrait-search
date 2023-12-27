import asyncio
from pathlib import Path

from tqdm import tqdm
from dependency_injector.wiring import Provide, inject

from portrait_search.data_source import BaseDataSource
from portrait_search.dependencies import Container
from portrait_search.portrait import PortraitService


@inject
async def generate_descriptions(
    local_data_folder: Path = Provide[Container.config.local_data_folder],
    data_sources: list[BaseDataSource] = Provide[Container.data_sources],
    portrait_service: PortraitService = Provide[Container.portrait_service],
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
    existing_hashes = await portrait_service.get_distinct_hashes()
    new_hashes = local_hashes - existing_hashes
    print(f"Found {len(new_hashes)} new portraits.")


if __name__ == "__main__":
    container = Container()
    container.init_resources()
    container.wire(modules=[__name__])

    asyncio.run(generate_descriptions())
