import secrets
import string
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import pytest
from aioresponses import aioresponses
from dependency_injector import providers
from mock import Mock

from portrait_search.core.mongodb import get_database
from portrait_search.dependencies import Container


@pytest.fixture
def temp_folder_path() -> Generator[Path, Any, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_responses() -> Generator[aioresponses, Any, Any]:
    with aioresponses() as m:
        yield m


@pytest.fixture
async def mongodb_database_for_test() -> AsyncGenerator[str, Any]:
    characters = string.ascii_letters + string.digits
    db_name = "test_database_" + "".join(secrets.choice(characters) for _ in range(10))

    yield db_name

    # TODO: enable this when I figure out which permission to set for a user to be able to drop a database
    # mongodb_connection: AsyncIOMotorClient = container.mongodb_connection()
    # await mongodb_connection.drop_database(db_name)


@pytest.fixture
def container() -> Container:
    container = Container()
    container.db.override(Mock())
    return container


@pytest.fixture
async def inject_mongodb_database_for_test(
    mongodb_database_for_test: str, container: Container
) -> AsyncGenerator[None, Any]:
    with container.db.override(
        providers.Resource(
            get_database,
            connection=Container.mongodb_connection,
            database_name=mongodb_database_for_test,
        )
    ):
        yield
