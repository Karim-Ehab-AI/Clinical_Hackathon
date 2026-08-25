import logging
from fastapi import APIRouter, Depends, status

from schemas.retrieval import SearchRequest, SearchResponse, RerankRequest, RerankResponse
from controllers.retrieval_controller import RetrievalController

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/retrieval", tags=["Retrieval"])


def get_retrieval_controller() -> RetrievalController:
    return RetrievalController()


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid search (dense + sparse) with RRF fusion and cross-encoder reranking.",
)
async def search_clinical_documents(
    request: SearchRequest,
    controller: RetrievalController = Depends(get_retrieval_controller),
) -> SearchResponse:
    """Perform hybrid retrieval on ingested clinical documents, returning ranked results with metadata provenance."""
    return await controller.search(request)


@router.post(
    "/rerank",
    response_model=RerankResponse,
    status_code=status.HTTP_200_OK,
    summary="Standalone cross-encoder reranking endpoint for custom query and candidate texts.",
)
async def rerank_documents(
    request: RerankRequest,
    controller: RetrievalController = Depends(get_retrieval_controller),
) -> RerankResponse:
    """Perform standalone cross-encoder reranking on arbitrary query and list of candidate documents."""
    return await controller.rerank(request)

