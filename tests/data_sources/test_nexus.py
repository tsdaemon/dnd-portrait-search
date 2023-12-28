from pathlib import Path
import shutil
import tempfile
from typing import Any, Generator
from aioresponses import aioresponses
import py7zr
import pytest
from portrait_search.core.config import Config

from portrait_search.data_sources.nexus import NexusDataSource
from portrait_search.portraits.entities import Portrait


@pytest.fixture
def nexus_data_source(temp_folder_path: Path) -> NexusDataSource:
    config = Config(
        NEXUSMODS_API_KEY="test_api_key", LOCAL_DATA_FOLDER=temp_folder_path
    )
    return NexusDataSource(
        config=config,
        url="https://www.nexusmods.com/pathfinderkingmaker/mods/9",
        game="pathfinderkingmaker",
        mod=9,
        file=1,
    )


@pytest.fixture
def mock_nexus_get_download_link_response(mock_responses: aioresponses) -> None:
    mock_responses.get(
        "https://api.nexusmods.com/v1/games/pathfinderkingmaker/mods/9/files/1/download_link.json",
        payload=[
            {
                "URI": "https://nexusmods.com/pathfinderkingmaker/mods/9?tab=files&file_id=1"
            },
            {},
        ],
    )


@pytest.fixture
def archive_fixture(fixtures_path: Path) -> Generator[Path, Any, Any]:
    test_mod_folder = fixtures_path / "nexus/test_mod"
    with tempfile.NamedTemporaryFile() as temp_file:
        with py7zr.SevenZipFile(temp_file.name, mode="w") as z:
            z.writeall(test_mod_folder, arcname="")
        yield Path(temp_file.name)


@pytest.fixture
def mock_nexus_download_response(
    mock_responses: aioresponses, archive_fixture: Path
) -> None:
    archive_content = archive_fixture.read_bytes()
    mock_responses.get(
        "https://nexusmods.com/pathfinderkingmaker/mods/9?tab=files&file_id=1",
        body=archive_content,
        headers={"content-length": str(len(archive_content))},
    )


@pytest.fixture
def expected_portraits(temp_folder_path: Path) -> list[Portrait]:
    root_folder = temp_folder_path / "nexusmods-pathfinderkingmaker-9-1"
    return [
        Portrait(
            fulllength_path=Path(
                root_folder / "some race/some type/AA-WI-MS-F107/Fulllength.png"
            ),
            medium_path=Path(
                root_folder / "some race/some type/AA-WI-MS-F107/Medium.png"
            ),
            small_path=Path(
                root_folder / "some race/some type/AA-WI-MS-F107/Small.png"
            ),
            base_path=temp_folder_path,
            tags=[
                "some race",
                "some type",
                "AA-WI-MS-F107",
            ],
            url="https://www.nexusmods.com/pathfinderkingmaker/mods/9",
        ),
        Portrait(
            fulllength_path=Path(
                root_folder / "some race/some type/AA-WI-MS-F107 copy/Fulllength.png"
            ),
            medium_path=Path(
                root_folder / "some race/some type/AA-WI-MS-F107 copy/Medium.png"
            ),
            small_path=Path(
                root_folder / "some race/some type/AA-WI-MS-F107 copy/Small.png"
            ),
            base_path=temp_folder_path,
            tags=[
                "some race",
                "some type",
                "AA-WI-MS-F107 copy",
            ],
            url="https://www.nexusmods.com/pathfinderkingmaker/mods/9",
        ),
        Portrait(
            fulllength_path=Path(
                root_folder / "another race/MX-IN-TL-F201/Fulllength.png"
            ),
            medium_path=Path(root_folder / "another race/MX-IN-TL-F201/Medium.png"),
            small_path=Path(root_folder / "another race/MX-IN-TL-F201/Small.png"),
            base_path=temp_folder_path,
            tags=[
                "another race",
                "MX-IN-TL-F201",
            ],
            url="https://www.nexusmods.com/pathfinderkingmaker/mods/9",
        ),
    ]


@pytest.mark.usefixtures(
    "mock_nexus_get_download_link_response", "mock_nexus_download_response"
)
@pytest.mark.asyncio
async def test_retrieve(
    nexus_data_source: NexusDataSource,
    temp_folder_path: Path,
    mock_responses: aioresponses,
    expected_portraits: list[Portrait],
) -> None:
    # GIVEN nothing is cached
    # WHEN call retrieve
    portraits = await nexus_data_source.retrieve(temp_folder_path)
    # THEN requests are made to get download link and download link
    mock_responses.assert_called_with(
        "https://api.nexusmods.com/v1/games/pathfinderkingmaker/mods/9/files/1/download_link.json",
        method="GET",
        headers={"apiKey": "test_api_key"},
    )
    mock_responses.assert_called_with(
        "https://nexusmods.com/pathfinderkingmaker/mods/9?tab=files&file_id=1",
        method="GET",
    )
    # THEN results are retrived
    assert portraits == expected_portraits


@pytest.fixture
def cached_archive(archive_fixture: Path, temp_folder_path: Path) -> None:
    shutil.copy(
        archive_fixture, temp_folder_path / "nexusmods-pathfinderkingmaker-9-1.7z"
    )


@pytest.mark.usefixtures("cached_archive")
@pytest.mark.asyncio
async def test_retrieve_cached_archive(
    nexus_data_source: NexusDataSource,
    temp_folder_path: Path,
    expected_portraits: list[Portrait],
) -> None:
    # GIVEN a mod archive is downloaded
    # WHEN call retrieve
    portraits = await nexus_data_source.retrieve(temp_folder_path)
    # THEN no requests are made and results are retrived
    assert portraits == expected_portraits


@pytest.fixture
def cached_extracted_archive(archive_fixture: Path, temp_folder_path: Path) -> None:
    with py7zr.SevenZipFile(archive_fixture, mode="r") as z:
        z.extractall(path=temp_folder_path / "nexusmods-pathfinderkingmaker-9-1")


@pytest.mark.usefixtures("cached_extracted_archive", "cached_archive")
@pytest.mark.asyncio
async def test_retrieve_cached_extracted_archive(
    nexus_data_source: NexusDataSource,
    temp_folder_path: Path,
    expected_portraits: list[Portrait],
) -> None:
    # GIVEN a mod archive is downloaded and extracted
    # WHEN call retrieve
    portraits = await nexus_data_source.retrieve(temp_folder_path)
    # THEN no requests are made and results are retrived
    assert portraits == expected_portraits
