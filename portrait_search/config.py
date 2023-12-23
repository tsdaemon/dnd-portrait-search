from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    mongodb_uri: str = Field(default=None, alias="MONGODB_URI")
    openai_api_key: str = Field(default=None, alias="OPENAI_API_KEY")
    nexusmods_api_key: str = Field(default=None, alias="NEXUSMODS_API_KEY")
    local_data_folder: Path = Field(default=None, alias="LOCAL_DATA_FOLDER")
