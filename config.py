"""Configuration settings for the 3GPP Telecom Spec Assistant."""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EVALUATION_DIR: Path = DATA_DIR / "evaluation"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    QDRANT_PATH: Path = STORAGE_DIR / "qdrant"
    RELATIONSHIPS_DIR: Path = STORAGE_DIR / "relationships"
    BM25_INDEX_PATH: Path = STORAGE_DIR / "bm25_index.pkl"

    # LLM Settings
    LLM_PROVIDER: Literal["gemini", "groq"] = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"  # or gemini-2.5-flash / gemini-1.5-pro
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 1024

    # Embedding & Vector Store
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_BACKEND: Literal["qdrant", "chroma"] = "qdrant"
    COLLECTION_NAME: str = "threegpp_specs"

    # Retrieval Hyperparameters
    TOP_K: int = 8
    FINAL_CONTEXT_K: int = 4
    RRF_K: int = 60
    MIN_RELEVANCE_SCORE: float = 0.25

    # Abstention Text
    ABSTENTION_MESSAGE: str = (
        "I could not find sufficient supporting evidence in the indexed 3GPP documents."
    )


settings = Settings()

# Ensure directories exist
for directory in [
    settings.DATA_DIR,
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.EVALUATION_DIR,
    settings.STORAGE_DIR,
    settings.QDRANT_PATH,
    settings.RELATIONSHIPS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
