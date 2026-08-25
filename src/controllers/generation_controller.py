import logging
from typing import Optional
from fastapi import HTTPException

from services.generation_service import GenerationService
from schemas.llm import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)


class GenerationController:
    """Controller orchestrating clinical RAG generation requests."""

    def __init__(self, generation_service: Optional[GenerationService] = None):
        self.generation_service = generation_service or GenerationService()

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        """Handle POST /api/v1/generation/generate request."""
        clean_query = req.query.strip()
        if not clean_query:
            raise HTTPException(status_code=400, detail="Query string cannot be empty.")

        try:
            return await self.generation_service.generate_response(query=clean_query)
        except Exception as e:
            logger.error(f"Generation controller failure: {e}")
            raise HTTPException(status_code=500, detail=f"Clinical Generation failed: {str(e)}")
