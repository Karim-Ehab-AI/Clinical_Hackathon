from fastapi import APIRouter, Depends, status
from controllers.generation_controller import GenerationController
from schemas.llm import GenerateRequest, GenerateResponse

router = APIRouter(prefix="/api/v1/generation", tags=["Clinical Generation"])


def get_controller() -> GenerationController:
    return GenerationController()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate concise First Aid emergency guidance with citation tracking",
    description=(
        "Executes end-to-end First Aid Clinical Decision Support pipeline. "
        "Retrieves candidates via hybrid search, applies 80% similarity threshold filtering, "
        "enforces scope & knowledge sufficiency AND-Gate guardrails, and generates concise bullet-point "
        "emergency instructions in the user's query language with exact chunk citations."
    ),
)
async def generate(
    req: GenerateRequest,
    controller: GenerationController = Depends(get_controller),
) -> GenerateResponse:
    """Generate concise First Aid emergency guidance with exact chunk citations."""
    return await controller.generate(req)
