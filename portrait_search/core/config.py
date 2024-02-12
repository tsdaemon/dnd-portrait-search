from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .enums import DistanceType, EmbedderType, SplitterType


class Config(BaseSettings):
    mongodb_uri: str = Field(default=None, alias="MONGODB_URI")
    mongodb_database_name: str = Field(default=None, alias="MONGODB_DATABASE_NAME")

    openai_api_key: str = Field(default=None, alias="OPENAI_API_KEY")

    local_data_folder: Path = Field(default=None, alias="LOCAL_DATA_FOLDER")

    data_sources_config_path: Path = Field(default=None, alias="DATA_SOURCES_CONFIG_PATH")
    nexusmods_api_key: str = Field(default=None, alias="NEXUSMODS_API_KEY")

    embedder_type: EmbedderType = Field(
        default=EmbedderType.INSTRUCTOR_LARGE_PATHFINDER_CHARACTER_INSTRUCTIONS, alias="EMBEDDER_TYPE"
    )
    splitter_type: SplitterType = Field(
        default=SplitterType.LANGCHAIN_RECURSIVE_TEXT_SPLITTER_CHUNK_160_OVERLAP_40, alias="SPLITTER_TYPE"
    )
    distance_type: DistanceType = Field(default=DistanceType.COSINE, alias="DISTANCE_TYPE")

    experiment: str = Field(default=None, alias="EXPERIMENT")

    model_config = SettingsConfigDict(str_strip_whitespace=True)
