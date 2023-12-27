from pathlib import Path
from pydantic import BaseModel
import imagehash  # type: ignore
from PIL import Image


class Portrait(BaseModel):
    """Represents an on-disk portrait representation."""

    fulllength_path: Path
    medium_path: Path
    small_path: Path
    tags: list[str]

    def get_fulllength_hash(self):
        # calculate hash if it doesn't exist
        hash_path = self.fulllength_path.with_suffix(".hash")
        if not hash_path.exists():
            image = Image.open(self.fulllength_path)
            image_hash = imagehash.average_hash(image)
            hash_path.write_text(str(image_hash))

        return hash_path.read_text()


class PortraitRecord(BaseModel):
    """Represents a portrait database record."""

    fulllength_path: str
    medium_path: str
    small_path: str
    tags: list[str]
    hash: str
    query: str
    description: str

    def from_portrait(self, portrait: Portrait):
        return PortraitRecord(
            fulllength_path=str(portrait.fulllength_path),
            medium_path=str(portrait.medium_path),
            small_path=str(portrait.small_path),
            tags=portrait.tags,
            hash=portrait.get_fulllength_hash(),
            query="",
            description="",
        )
