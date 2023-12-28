from pathlib import Path
import tempfile
from typing import Any, Generator
from aioresponses import aioresponses

import pytest


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
