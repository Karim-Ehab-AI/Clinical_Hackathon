import logging
from typing import List

from interfaces.reranker import Reranker
from schemas.retrieval import RerankedDocument

logger = logging.getLogger(__name__)


class NoOpReranker(Reranker):
    """Fallback / Pass-through Reranker provider.
    Returns incoming candidate list unchanged (retaining fused score and order).
    """

    async def rerank(
        self, query: str, documents: List[RerankedDocument]
    ) -> List[RerankedDocument]:
        """Pass through candidate documents without modifying scores or order."""
        logger.info(f"ℹ️ NoOpReranker active: returning {len(documents)} candidates with fused RRF ranking.")
        return list(documents)
