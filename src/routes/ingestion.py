from fastapi import APIRouter, UploadFile, File, Depends
from controllers.ingestion_controller import IngestionController
from services.document_service import DocumentService
from services.vector_store_service import VectorStoreService
from providers.docling_provider import DoclingProvider
from providers.colab_embedding_provider import ColabEmbeddingProvider
from providers.qdrant_provider import QdrantProvider
from schemas.ingestion import IngestionResponse

router = APIRouter(prefix="/api/v1/ingestion", tags=["Ingestion Pipeline"])


def get_ingestion_controller() -> IngestionController:
    """Dependency Provider creating the IngestionController with injected interfaces."""
    parser = DoclingProvider()
    embedding_provider = ColabEmbeddingProvider()
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
    description="Parses PDF via Docling, cleans page furniture, performs structure-aware chunking with BAAI/bge-m3 tokenizer, extracts NICE/ESC metadata, generates dense+sparse embeddings via Colab, and stores named-vector points in Qdrant.",
)
async def upload_clinical_pdf(
    file: UploadFile = File(...),
    controller: IngestionController = Depends(get_ingestion_controller),
) -> IngestionResponse:
    return await controller.handle_pdf_upload(file)
