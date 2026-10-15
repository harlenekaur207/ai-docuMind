from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ai_provider: str = "auto"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    chroma_path: Path = Path("./data/chroma")
    upload_path: Path = Path("./data/uploads")
    max_file_size_mb: int = 20
    top_k: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def resolved_provider(self) -> str:
        if self.ai_provider == "auto":
            return "openai" if self.openai_api_key else "ollama"
        return self.ai_provider


settings = Settings()
