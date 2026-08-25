from abc import ABC, abstractmethod
from typing import List
from schemas.documents import EmbeddingResult


class EmbeddingProvider(ABC):
    """Abstract Interface for Embedding Providers generating unified dense + sparse vectors."""

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a list of text strings returning dense and sparse representations per text."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query string returning dense and sparse representations."""
        pass

    @abstractmethod
    async def check_health(self) -> bool:
        """Check the operational readiness of the remote embedding service."""
        pass
