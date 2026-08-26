import logging
from fastapi import APIRouter, UploadFile, File, Depends
from controllers.ingestion_controller import IngestionController
from services.document_service import DocumentService
from services.vector_store_service import VectorStoreService
from providers.local_embedding_provider import LocalEmbeddingProvider
from providers.colab_embedding_provider import ColabEmbeddingProvider
from providers.local_docling_provider import LocalDoclingProvider
from providers.docling_provider import DoclingProvider
from providers.qdrant_provider import QdrantProvider
from schemas.ingestion import IngestionResponse
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingestion", tags=["Ingestion Pipeline"])


def get_ingestion_controller() -> IngestionController:
    """Dependency Provider creating the IngestionController with injected interfaces."""
    # Choose Embedding Provider based on settings
    if settings.EMBEDDING_PROVIDER_TYPE == "remote" and settings.EMBEDDING_API_URL:
        logger.info("Injecting ColabEmbeddingProvider for ingestion route")
        embedding_provider = ColabEmbeddingProvider()
    else:
        logger.info("Injecting LocalEmbeddingProvider (AWS/in-process) for ingestion route")
        embedding_provider = LocalEmbeddingProvider()

    # Choose Docling Parser Provider based on settings
    if settings.DOCLING_PROVIDER_TYPE == "remote" and settings.EMBEDDING_API_URL:
        parser = DoclingProvider()
    else:
        parser = LocalDoclingProvider()

    qdrant_provider = QdrantProvider()
    vector_store_service = VectorStoreService(vector_store=qdrant_provider)

    document_service = DocumentService(
        parser=parser,
        embedding_provider=embedding_provider,
        vector_store_service=vector_store_service,
    )
    return IngestionController(document_service=document_service)


@router.post(
    "/upload",
    response_model=IngestionResponse,
    summary="Upload and ingest a clinical PDF document",
    description="Parses PDF via Docling, cleans page furniture, performs structure-aware chunking with BAAI/bge-m3 tokenizer, extracts metadata, generates dense+sparse embeddings locally or via remote, and stores points in Qdrant.",
)
@router.post(
    "/ingest-pdf",
    response_model=IngestionResponse,
    include_in_schema=False,
)
async def upload_clinical_pdf(
    file: UploadFile = File(...),
    controller: IngestionController = Depends(get_ingestion_controller),
) -> IngestionResponse:
    return await controller.handle_pdf_upload(file)

