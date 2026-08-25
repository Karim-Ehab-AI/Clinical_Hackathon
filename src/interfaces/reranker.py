from abc import ABC, abstractmethod
from typing import List
from schemas.retrieval import RerankedDocument


class Reranker(ABC):
    """Abstract Interface for cross-encoder / reranking providers."""

    @abstractmethod
    async def rerank(
        self, query: str, documents: List[RerankedDocument]
    ) -> List[RerankedDocument]:
        """Rerank candidates in response to query, returning reranked documents with updated scores."""
        pass
