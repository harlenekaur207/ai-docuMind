from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_path: Path = Path("./data/chroma")
    upload_path: Path = Path("./data/uploads")
    max_file_size_mb: int = 20
    top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

