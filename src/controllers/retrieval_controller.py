import logging
from typing import Optional
from fastapi import HTTPException, status

from schemas.retrieval import SearchRequest, SearchResponse
from services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RetrievalController:
    """Controller handling retrieval search endpoints."""

    def __init__(self, retrieval_service: Optional[RetrievalService] = None):
        self.retrieval_service = retrieval_service or RetrievalService()

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Handle search request validation and invoke RetrievalService."""
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query string cannot be empty.",
            )

        try:
            return await self.retrieval_service.search(query=request.query)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during clinical search retrieval: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Clinical retrieval failed: {str(e)}",
            )

    async def rerank(self, request: "RerankRequest") -> "RerankResponse":
        """Handle standalone rerank request validation and invoke RetrievalService."""
        from schemas.retrieval import RerankRequest, RerankResponse
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query string cannot be empty.",
            )
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Documents list cannot be empty.",
            )

        try:
            return await self.retrieval_service.rerank_standalone(
                query=request.query, documents=request.documents
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during standalone reranking: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Reranking failed: {str(e)}",
            )

