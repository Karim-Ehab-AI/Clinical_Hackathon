from typing import Optional
from pydantic import BaseModel


class IngestionResponse(BaseModel):
    """Response returned upon completion of file ingestion upload."""

    status: str  # "success", "already_exists", "error"
    document_id: str
    filename: str
    chunks_created: int
    vectors_stored: int
    message: Optional[str] = None
