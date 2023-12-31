from pathlib import Path

import imagehash  # type: ignore
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from portrait_search.core.mongodb import PyObjectId


class PortraitRecord(BaseModel):
    """Represents a portrait database record."""

    id: PyObjectId | None = Field(alias="_id", default=None)

    fulllength_path: str
    medium_path: str
    small_path: str
    tags: list[str]
    url: str
    hash: str
    query: str
    description: str

    model_config = ConfigDict(populate_by_name=True)


class Portrait(BaseModel):
    """Represents an on-disk local portrait."""

    fulllength_path: Path
    medium_path: Path
    small_path: Path
    base_path: Path
    tags: list[str]
    url: str

    def get_fulllength_hash(self) -> str:
        # calculate hash if it doesn't exist
        hash_path = self.fulllength_path.with_suffix(".hash")
        if not hash_path.exists():
            image = Image.open(self.fulllength_path)
            image_hash = imagehash.average_hash(image)
            hash_path.write_text(str(image_hash))

        return hash_path.read_text()

    def to_record(self) -> PortraitRecord:
        return PortraitRecord(
            fulllength_path=str(self.fulllength_path.relative_to(self.base_path)),
            medium_path=str(self.medium_path.relative_to(self.base_path)),
            small_path=str(self.small_path.relative_to(self.base_path)),
            tags=self.tags,
            hash=self.get_fulllength_hash(),
            url=self.url,
            query="",
            description="",
        )
