"""Configuration settings for the 3GPP RAG Assistant."""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment variable configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base project paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EVALUATION_DIR: Path = DATA_DIR / "evaluation"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    QDRANT_PATH: Path = STORAGE_DIR / "qdrant"

    # LLM inference settings
    LLM_PROVIDER: Literal["groq", "gemini"] = "groq"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-latest"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 1024

    # Embedding model and vector store
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_BACKEND: Literal["qdrant", "chroma"] = "qdrant"
    COLLECTION_NAME: str = "3gpp_specs"
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    # Cross-encoder re-ranker settings
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CANDIDATE_K: int = 15
    FINAL_CONTEXT_K: int = 4

    # Retrieval and evidence gate thresholds
    TOP_K: int = 4
    RRF_K: int = 60
    MIN_RELEVANCE_SCORE: float = 0.40

    # Default fallback message when evidence is insufficient
    ABSTENTION_MESSAGE: str = (
        "I could not find sufficient supporting evidence in the indexed 3GPP documents."
    )


# Instantiate global settings object
settings = Settings()

# Automatically ensure required project directories exist
for directory in [
    settings.DATA_DIR,
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.EVALUATION_DIR,
    settings.STORAGE_DIR,
    settings.QDRANT_PATH,
]:
    directory.mkdir(parents=True, exist_ok=True)
