from pathlib import Path
from pydantic import BaseModel


class Portrait(BaseModel):
    fulllength_path: Path
    medium_path: Path
    small_path: Path
    tags: list[str]
