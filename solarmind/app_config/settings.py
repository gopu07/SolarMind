from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os
import structlog

log = structlog.get_logger(__name__)

def mask_secret(value: Optional[str]) -> str:
    """Mask sensitive values for safe logging/display."""
    if not value or len(value) < 8:
        return "NOT SET" if not value else "****"
    return value[:4] + "****" + value[-4:]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core API Keys
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # Infrastructure
    database_url: Optional[str] = None
    environment: str = "development"
    
    # Paths
    model_path: str = "./models/artifacts/model.pkl"
    rag_index_path: str = "./rag/chroma_store"
    
    # Logging
    log_level: str = "INFO"

settings = Settings()

def print_config_status():
    """Safety print configuration status to logs on startup."""
    print("\n" + "="*40)
    print("[CONFIG STATUS]")
    print(f"Environment: {settings.environment}")
    print(f"OpenAI API Key: {'configured' if settings.openai_api_key else 'NOT SET'}")
    print(f"Gemini API Key: {'configured' if settings.gemini_api_key else 'NOT SET'}")
    print(f"Database URL: {'configured' if settings.database_url else 'NOT SET'}")
    print(f"Model Path: {settings.model_path}")
    print(f"RAG Index: {settings.rag_index_path}")
    print("="*40 + "\n")
    
    if not settings.gemini_api_key and settings.environment != "test":
        log.warning("GEMINI_API_KEY_NOT_CONFIGURED", hint="Check your .env file or environment variables.")
    if not settings.openai_api_key and settings.environment != "test":
        log.warning("OPENAI_API_KEY_NOT_CONFIGURED", hint="Check your .env file or environment variables.")

def get_masked_settings() -> dict:
    """Return settings with masked secrets for API consumption."""
    return {
        "environment": settings.environment,
        "openai_api_key": mask_secret(settings.openai_api_key),
        "gemini_api_key": mask_secret(settings.gemini_api_key),
        "database_url": "configured" if settings.database_url else "NOT SET",
        "model_path": "configured" if settings.model_path else "NOT SET",
        "rag_index_path": "configured" if settings.rag_index_path else "NOT SET"
    }
