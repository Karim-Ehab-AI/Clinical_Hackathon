import logging
from typing import List, Optional

from interfaces.reranker import Reranker
from providers.colab_reranker_provider import ColabReranker
from providers.noop_reranker_provider import NoOpReranker
from schemas.retrieval import RerankedDocument
from config.settings import settings

logger = logging.getLogger(__name__)


class RerankingService:
    """Service wrapper for candidate reranking with graceful fallback to NoOpReranker."""

    def __init__(self, reranker: Optional[Reranker] = None):
        if reranker is not None:
            self.primary_reranker = reranker
        elif settings.RERANKER_ENABLED and settings.RERANKER_API_URL:
            self.primary_reranker = ColabReranker()
        else:
            logger.info("Reranker is disabled or API URL is missing. Defaulting to NoOpReranker.")
            self.primary_reranker = NoOpReranker()

        self.fallback_reranker = NoOpReranker()

    async def rerank(
        self, query: str, documents: List[RerankedDocument]
    ) -> List[RerankedDocument]:
        """Rerank candidate documents.
        
        If cross-encoder succeeds, explicitly sort descending by reranker score.
        If cross-encoder fails/disabled, fallback to NoOp maintaining incoming RRF fusion rank.
        """
        if not documents:
            return []

        try:
            results = await self.primary_reranker.rerank(query, documents)
            # If a cross-encoder (non-NoOp) provider was used, explicitly sort by new score
            if not isinstance(self.primary_reranker, NoOpReranker):
                results.sort(key=lambda doc: doc.score, reverse=True)
                logger.info(f"Reranking completed and explicitly sorted {len(results)} items by cross-encoder score.")
            return results

        except Exception as e:
            logger.warning(f"⚠️ Primary reranker failed ({e}). Falling back to NoOpReranker (RRF order).")
            # Fallback path: keeps existing RRF-fusion order
            return await self.fallback_reranker.rerank(query, documents)
