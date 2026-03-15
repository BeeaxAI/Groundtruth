"""
Phase 1: Configuration Management
Centralized, validated configuration using pydantic-settings.
Supports .env files, environment variables, and defaults.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings with validation."""

    # --- Gemini / Google AI ---
    google_api_key: str = Field(default="", description="Gemini API key")
    gemini_live_model: str = Field(
        default="gemini-2.5-flash-native-audio-preview-12-2025",
        description="Model for Live API (real-time audio/video)",
    )
    gemini_text_model: str = Field(
        default="gemini-2.5-flash-preview-05-20",
        description="Model for text-only queries (REST fallback)",
    )
    embedding_model: str = Field(
        default="text-embedding-004",
        description="Model for generating document embeddings (Super Memory)",
    )
    gemini_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    gemini_max_output_tokens: int = Field(default=2048, ge=1)

    # --- Google Cloud ---
    google_cloud_project: str = Field(default="", description="GCP project ID")
    gcs_bucket_name: str = Field(default="", description="GCS bucket for documents")
    google_genai_use_vertexai: bool = Field(default=False)

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    debug: bool = Field(default=False)
    cors_origins: list[str] = Field(default=["*"])

    # --- Document Processing ---
    chunk_size: int = Field(default=500, ge=100, le=5000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)
    max_document_size_mb: float = Field(default=20.0)
    max_documents: int = Field(default=50)
    max_retrieval_chunks: int = Field(default=5)
    max_context_chars: int = Field(default=4000)

    # --- Audio ---
    input_sample_rate: int = Field(default=16000)
    output_sample_rate: int = Field(default=24000)
    voice_name: str = Field(default="Kore")

    # --- Security ---
    max_query_length: int = Field(default=5000)
    rate_limit_rpm: int = Field(default=30)
    enable_audit_log: bool = Field(default=True)

    # --- Paths ---
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    frontend_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "frontend")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v, info):
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Singleton
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            _settings = Settings(_env_file=str(env_path))
        else:
            _settings = Settings()
    return _settings
