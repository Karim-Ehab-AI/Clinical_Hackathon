import logging
from typing import Tuple, Optional
from interfaces.document_parser import DocumentParser
from interfaces.embedding_provider import EmbeddingProvider
from services.storage_service import StorageService
from services.cleaning_service import CleaningService
from services.chunking_service import ChunkingService
from services.vector_store_service import VectorStoreService
from services.pdf_chunking_pipeline import PDFChunkingPipeline
from providers.qdrant_provider import QdrantProvider
from schemas.ingestion import IngestionResponse
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """High-Level Document Pipeline Service orchestrating document ingestion end-to-end."""

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        storage_service: Optional[StorageService] = None,
        cleaning_service: Optional[CleaningService] = None,
        chunking_service: Optional[ChunkingService] = None,
        pdf_pipeline: Optional[PDFChunkingPipeline] = None,
    ):
        self.parser = parser
        self.embedding_provider = embedding_provider
        self.vector_store_service = vector_store_service
        self.storage_service = storage_service or StorageService()
        self.cleaning_service = cleaning_service or CleaningService()
        self.chunking_service = chunking_service or ChunkingService()
        self.pdf_pipeline = pdf_pipeline or PDFChunkingPipeline()
        self.qdrant_provider = QdrantProvider()

    async def process_pdf(self, file_name: str, content: bytes) -> IngestionResponse:
        """Run full PDF ingestion pipeline (Local in-process on AWS or remote fallback)."""
        # Step 1: Save file to src/assets/{file_hash}.pdf and check deduplication
        file_hash, file_path, file_exists = self.storage_service.save_file(content)

        # Check if already ingested in Vector Store
        if file_exists and self.qdrant_provider.document_exists(file_hash):
            logger.info(f"Document {file_name} (hash: {file_hash}) already ingested. Returning early response.")
            return IngestionResponse(
                status="already_exists",
                document_id=file_hash,
                filename=file_name,
                chunks_created=0,
                vectors_stored=0,
                message="Document already ingested in assets storage and vector store.",
            )

        # Step 2: Run PDF Chunking & Embedding Pipeline (Local or Remote)
        logger.info(
            f"Running PDF chunking & embedding pipeline for: {file_name} (ID: {file_hash}) "
            f"[Mode: {settings.DOCLING_PROVIDER_TYPE}/{settings.EMBEDDING_PROVIDER_TYPE}]"
        )
        
        chunks = await self.pdf_pipeline.process_pdf(
            pdf_path=file_path,
            embedding_provider=self.embedding_provider,
        )

        if not chunks:
            raise ValueError(f"No structural chunks returned from PDF chunking pipeline for '{file_name}'.")

        # Step 3: Upsert resulting DocumentChunk objects directly into Qdrant
        logger.info(f"Upserting {len(chunks)} DocumentChunk points into Qdrant Vector DB...")
        stored_count = self.qdrant_provider.upsert_document_chunks(chunks)

        return IngestionResponse(
            status="success",
            document_id=file_hash,
            filename=file_name,
            chunks_created=len(chunks),
            vectors_stored=stored_count,
            message="PDF ingestion pipeline completed successfully.",
        )
