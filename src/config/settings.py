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

    # Embedding Provider (Colab BGE-M3 HTTP Service)
    EMBEDDING_API_URL: str = "http://localhost:8000"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_TIMEOUT: int = 30
    EMBEDDING_BATCH_SIZE: int = 32

    # Storage Settings
    ASSETS_DIR: str = "src/assets"

    # Retrieval & Search Settings
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20
    HYBRID_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    RRF_K: int = 60

    # Reranker Provider Settings (Colab BGE Cross-Encoder Service)
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

