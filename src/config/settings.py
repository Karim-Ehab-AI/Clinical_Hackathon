import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Absolute path to project root where .env lives
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Application Settings."""

    APP_NAME: str = "clinical-rag"
    DEBUG: bool = True

    # Qdrant Settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "clinical_documents"

    # Embedding Provider Configuration (Default: Local in-process BGE-M3 for AWS)
    EMBEDDING_PROVIDER_TYPE: str = "local"  # "local" | "remote"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_DEVICE: str = "auto"  # "auto" | "cuda" | "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_TIMEOUT: int = 30
    EMBEDDING_API_URL: str = ""  # Used when EMBEDDING_PROVIDER_TYPE="remote"
    EMBEDDING_API_KEY: str = ""

    # Docling & PDF Processing
    DOCLING_PROVIDER_TYPE: str = "local"  # "local" | "remote"
    DOCLING_DO_OCR: bool = False

    # Storage Settings
    ASSETS_DIR: str = "src/assets"

    # Retrieval & Search Settings
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    HYBRID_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    RRF_K: int = 60

    # Reranker Settings (Disabled by default)
    RERANKER_ENABLED: bool = False
    RERANKER_API_URL: str = ""
    RERANKER_API_KEY: str = ""
    RERANKER_TIMEOUT: int = 30

    # Gemini LLM Provider Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    MIN_SIMILARITY_SCORE_THRESHOLD: float = 80.0

    class Config:
        env_file = str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else ".env"
        extra = "ignore"


settings = Settings()
