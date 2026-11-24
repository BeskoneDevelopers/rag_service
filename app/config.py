from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    database_sync_url: str
    debug: bool = False
    jwt_secret: SecretStr = Field(min_length=32)
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    celery_task_time_limit: int = Field(default=600, gt=0)
    celery_visibility_timeout: int = Field(default=900, gt=0)

    llm_chat_url: str = "http://host.docker.internal:8081/v1/chat/completions"
    llm_model: str = "qwen3.5-2b"
    llm_timeout_seconds: int = Field(default=120, gt=0)
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_generation_timeout_seconds: float = Field(default=60, gt=0)

    embeddings_url: str = "http://host.docker.internal:8082/v1/embeddings"
    embeddings_model: str = "bge-m3"
    embeddings_batch_size: int = Field(default=8, ge=1)
    embeddings_timeout_seconds: int = Field(default=180, gt=0)

    @model_validator(mode="after")
    def validate_visibility_timeout(self):
        if self.celery_visibility_timeout <= self.celery_task_time_limit:
            raise ValueError("CELERY_VISIBILITY_TIMEOUT must exceed CELERY_TASK_TIME_LIMIT")
        return self

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
