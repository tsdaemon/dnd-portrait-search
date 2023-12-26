import asyncio
from portrait_search.plt.config import Config
from portrait_search.data_sources import sources_from_yaml


async def generate_descriptions():
    print("Extracting portraits from sources...")
    sources = sources_from_yaml()
    portraits = []
    local_portraits_folder = Config().local_data_folder / "portraits"
    for source in sources:
        portraits.extend(await source.retrieve(local_portraits_folder))

    print(portraits)


if __name__ == "__main__":
    asyncio.run(generate_descriptions())
