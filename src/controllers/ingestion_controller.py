import logging
from fastapi import UploadFile, HTTPException, status
from services.document_service import DocumentService
from schemas.ingestion import IngestionResponse

logger = logging.getLogger(__name__)


class IngestionController:
    """Controller handling ingestion request validation and service invocation."""

    def __init__(self, document_service: DocumentService):
        self.document_service = document_service

    async def handle_pdf_upload(self, file: UploadFile) -> IngestionResponse:
        """Validate PDF upload file and delegate to DocumentService."""
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Only PDF files are supported.",
            )

        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF file is empty.",
            )

        try:
            response = await self.document_service.process_pdf(
                file_name=file.filename,
                content=content,
            )
            return response
        except ValueError as ve:
            logger.error(f"Ingestion processing error: {ve}")
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
        except RuntimeError as re:
            logger.error(f"Ingestion service error: {re}")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(re))
        except Exception as e:
            logger.error(f"Unexpected ingestion failure: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred during PDF ingestion: {str(e)}",
            )
