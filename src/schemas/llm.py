from typing import Optional, List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation metadata identifying a chunk used in the clinical response."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    source: Optional[str] = Field(default=None, description="Source PDF filename.")
    document_id: Optional[str] = Field(default=None, description="Unique document ID.")
    recommendation_id: Optional[str] = Field(default=None, description="Clinical recommendation ID if present.")
    pdf_page: Optional[int] = Field(default=None, description="PDF page number.")
    section: Optional[str] = Field(default=None, description="Document section header.")
    source_text: Optional[str] = Field(default=None, description="Original text content of the cited chunk.")
    score: Optional[float] = Field(default=None, description="Similarity score (0.0 to 1.0).")
    percentage_score: Optional[float] = Field(default=None, description="Similarity percentage score (e.g. 88.5%).")



class ClinicalLLMResponse(BaseModel):
    """Structured Pydantic Output schema for Clinical LLM Generation."""

    is_in_scope: bool = Field(
        ...,
        description="True if the query is strictly about First Aid or Emergency management. False for other topics."
    )
    is_knowledge_sufficient: bool = Field(
        ...,
        description="True if the provided context contains enough accurate information to confidently answer the query. False otherwise."
    )
    answer: Optional[str] = Field(
        default=None,
        description="The concise first-aid answer in bullet points with citations. MUST be None if either flag is False."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of citations for candidate chunks actually used to construct the answer. MUST be empty if answer is None."
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Polite explanation if the request is out of scope or context is insufficient."
    )
    provider: str = Field(default="gemini", description="The LLM provider name used.")
    model_name: str = Field(default="gemini-3.1-flash-lite", description="Model name used for generation.")
    filtered_chunks_count: int = Field(default=0, description="Number of context chunks passing the >=80% threshold.")


class GenerateRequest(BaseModel):
    """Input request schema for clinical generation API."""

    query: str = Field(..., min_length=2, description="Clinical query string.")


class GenerateResponse(BaseModel):
    """API Response schema for clinical generation API."""

    query: str
    result: ClinicalLLMResponse
    retrieved_chunks_count: int = Field(default=0, description="Total chunks retrieved before threshold filtering.")
    filtered_chunks_count: int = Field(default=0, description="Chunks passing >=80% similarity threshold (max 3 passed to LLM).")
