import abc
from enum import Enum
import os
from pathlib import Path
from typing import Annotated, Literal
import aiohttp
import aiofiles
import py7zr
from pydantic import BaseModel, Field, TypeAdapter
import tqdm

from portrait_search.config import Config
from portrait_search.entities.portrait import Portrait
import yaml


class SourceKind(Enum):
    NexusMods = "nexusmods"


class Source(BaseModel):
    kind: str

    @abc.abstractmethod
    async def retrieve(self, folder: Path) -> list[Portrait]:
        raise NotImplementedError()


class SourceError(Exception):
    pass


class NexusSource(Source):
    kind: Literal["nexusmods"]
    game: str
    mod: int
    file: int

    async def retrieve(self, portraits_data_folder: Path) -> list[Portrait]:
        # Get mod if not already downloaded
        nexus_key = Config().nexusmods_api_key
        download_url = await self._get_download_url(nexus_key)
        archive_file = portraits_data_folder / f"{self._get_folder_name()}.7z"
        if not archive_file.exists():
            await self._download_archive_locally(download_url, archive_file)
        # Extract mod if not already extracted
        extraction_folder = portraits_data_folder / self._get_folder_name()
        if not extraction_folder.exists():
            print(f"Unpacking {archive_file} to {extraction_folder}")
            with py7zr.SevenZipFile(archive_file, mode="r") as z:
                z.extractall(path=extraction_folder)
        # Parse result to portraits
        return self._parse_result(extraction_folder)

    def _get_folder_name(self) -> str:
        return f"{self.kind}-{self.game}-{self.mod}-{self.file}"

    async def _get_download_url(self, nexus_key: str) -> str:
        url = f"https://api.nexusmods.com/v1/games/{self.game}/mods/{self.mod}/files/{self.file}/download_link.json"

        async with aiohttp.ClientSession() as session:
            async with session.get(url=url, headers={"apiKey": nexus_key}) as response:
                response.raise_for_status()
                result: list = await response.json()

        if len(result) == 0:
            raise SourceError("NexusMods download link response is empty")

        return result[0]["URI"]

    async def _download_archive_locally(self, download_url: str, archive: Path) -> None:
        async with aiofiles.open(archive, mode="wb") as f:
            async with aiohttp.ClientSession() as session:
                async with session.get(url=download_url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))

                    tqdm_params = {
                        "desc": f"Downloading mod: {download_url}",
                        "total": total,
                        "miniters": 1,
                        "unit": "B",
                        "unit_scale": True,
                        "unit_divisor": 1024,
                    }
                    with tqdm.tqdm(**tqdm_params) as pb:  # type: ignore[call-overload]
                        async for chunk in response.content.iter_chunked(8196):
                            await f.write(chunk)
                            pb.update(len(chunk))

    def _parse_result(self, extract_path: Path) -> list[Portrait]:
        result = []
        for root, _, files in os.walk(extract_path):
            if set(files) != {"Fulllength.png", "Medium.png", "Small.png"}:
                continue
            root_path = Path(root)
            # get relative path from extract_path to root
            internal_path = root_path.relative_to(extract_path)
            result.append(
                Portrait(
                    fulllength_path=root_path / "fulllength.png",
                    medium_path=root_path / "medium.png",
                    small_path=root_path / "small.png",
                    tags=list(internal_path.parts),
                )
            )

        return result


SourceType = Annotated[NexusSource, Field(discriminator="kind")]


def sources_from_yaml() -> list[Source]:
    # get path of sources.yaml
    sources_config = Path(__file__).parent / "sources.yaml"

    with open(sources_config, "r") as file:
        sources_data = yaml.safe_load(file)

    sources: list[Source] = []
    for source_data in sources_data["sources"]:
        adapter = TypeAdapter(SourceType)
        source = adapter.validate_python(source_data)
        sources.append(source)  # type: ignore

    return sources
