from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    mongodb_uri: str = Field(default=None, alias="MONGODB_URI")
    mongodb_database_name: str = Field(default=None, alias="MONGODB_DATABASE_NAME")

    openai_api_key: str = Field(default=None, alias="OPENAI_API_KEY")

    local_data_folder: Path = Field(default=None, alias="LOCAL_DATA_FOLDER")

    data_sources_config_path: Path = Field(default=None, alias="DATA_SOURCES_CONFIG_PATH")
    nexusmods_api_key: str = Field(default=None, alias="NEXUSMODS_API_KEY")

    model_config = SettingsConfigDict(str_strip_whitespace=True)
