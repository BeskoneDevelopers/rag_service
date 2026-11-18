from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    database_sync_url: str
    debug: bool = False

    llm_chat_url: str = "http://host.docker.internal:8081/v1/chat/completions"
    llm_model: str = "qwen3.5-2b"
    llm_timeout_seconds: int = Field(default=120, gt=0)

    embeddings_url: str = "http://host.docker.internal:8082/v1/embeddings"
    embeddings_model: str = "bge-m3"
    embeddings_batch_size: int = Field(default=8, ge=1)
    embeddings_timeout_seconds: int = Field(default=180, gt=0)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
