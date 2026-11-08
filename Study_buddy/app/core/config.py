"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared across the application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "StudyBuddy AI"
    database_url: str = "sqlite:///./studybuddy.db"
    chroma_path: str = "./chroma_data"
    secret_key: str = Field(
        default="replace-this-development-secret-before-deployment",
        min_length=32,
    )
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    access_token_expire_minutes: int = Field(default=60, gt=0)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    tavily_api_key: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_upload_size_mb: int = Field(default=20, gt=0)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    topic_cluster_count: int = Field(default=5, gt=0)
    cache_ttl_seconds: int = Field(default=300, gt=0)
    cache_max_entries: int = Field(default=512, gt=0)
    max_message_length: int = Field(default=4000, gt=0)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Ensure overlapping chunks always advance during splitting."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
