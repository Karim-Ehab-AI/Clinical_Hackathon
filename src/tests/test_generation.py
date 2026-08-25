import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from main import app
from schemas.llm import ClinicalLLMResponse, Citation
from schemas.retrieval import SearchResponse, SearchResult
from services.generation_service import GenerationService
from providers.gemini_llm_provider import GeminiLLMProvider


@pytest.mark.asyncio
async def test_score_threshold_filtering_refusal():
    """Test that if 0 chunks pass >= 80% threshold, service returns refusal without calling Gemini."""
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search = AsyncMock(
        return_value=SearchResponse(
            query="Out of scope query",
            results=[
                SearchResult(
                    text="Low relevant text",
                    score=0.5,
                    percentage_score=50.0,  # Below 80% threshold
                    document_id="doc1",
                    source="source1.pdf",
                    pdf_page=1,
                    document_page=1,
                    section="Sec 1",
                    recommendation_id=None,
                    is_table=False,
                )
            ]
        )
    )

    mock_llm_provider = MagicMock(spec=GeminiLLMProvider)
    mock_llm_provider.generate = AsyncMock()

    service = GenerationService(
        retrieval_service=mock_retrieval_service,
        llm_provider=mock_llm_provider,
        min_score_threshold=80.0,
    )

    response = await service.generate_response("Out of scope query")

    # Assert LLM was NOT called
    mock_llm_provider.generate.assert_not_called()

    # Assert refusal response
    assert response.result.is_knowledge_sufficient is False
    assert response.result.answer is None
    assert response.result.citations == []
    assert "insufficient" in response.result.refusal_reason.lower() or "غير كافية" in response.result.refusal_reason
    assert response.filtered_chunks_count == 0
    assert response.retrieved_chunks_count == 1


@pytest.mark.asyncio
async def test_top_3_chunks_selection_and_no_citation_leakage():
    """Test that max 3 chunks pass to Gemini and unused context chunks do not leak into citations."""
    mock_retrieval_service = MagicMock()

    # Create 5 high-scoring search results (>= 80%)
    high_scoring_results = [
        SearchResult(
            text=f"Clinical chunk text {i}",
            score=0.9,
            percentage_score=90.0,
            document_id=f"doc_chunk_{i}",
            source="nice_guidelines.pdf",
            pdf_page=i,
            document_page=i,
            section=f"Section {i}",
            recommendation_id=f"REC-1.{i}",
            is_table=False,
        )
        for i in range(1, 6)  # 5 items
    ]

    mock_retrieval_service.search = AsyncMock(
        return_value=SearchResponse(
            query="Diabetes treatment",
            results=high_scoring_results
        )
    )

    mock_llm_provider = MagicMock(spec=GeminiLLMProvider)
    # Gemini returns answer using ONLY chunk 1 and chunk 2, omitting chunk 3!
    mock_llm_provider.generate = AsyncMock(
        return_value=ClinicalLLMResponse(
            is_in_scope=True,
            is_knowledge_sufficient=True,
            answer="Metformin and lifestyle modifications are recommended.",
            citations=[
                Citation(chunk_id="doc_chunk_1", recommendation_id="REC-1.1", pdf_page=1, section="Section 1"),
                Citation(chunk_id="doc_chunk_2", recommendation_id="REC-1.2", pdf_page=2, section="Section 2"),
            ],
            refusal_reason=None,
            provider="gemini",
            model_name="gemini-1.5-flash",
            filtered_chunks_count=3,
        )
    )

    service = GenerationService(
        retrieval_service=mock_retrieval_service,
        llm_provider=mock_llm_provider,
        min_score_threshold=80.0,
    )

    response = await service.generate_response("Diabetes treatment")

    # 1. Assert Gemini received ONLY Top 3 chunks (out of 5 high scoring)
    mock_llm_provider.generate.assert_called_once()
    _, kwargs = mock_llm_provider.generate.call_args
    passed_docs = kwargs["filtered_docs"]
    assert len(passed_docs) == 3

    # 2. Assert NO citation leakage: Chunk 3 was in context, but NOT cited by Gemini!
    cited_ids = [c.chunk_id for c in response.result.citations]
    assert len(cited_ids) == 2
    assert "doc_chunk_1" in cited_ids
    assert "doc_chunk_2" in cited_ids
    assert "doc_chunk_3" not in cited_ids  # NO LEAKAGE!


@pytest.mark.asyncio
async def test_and_gate_failure_clears_citations():
    """Test that if AND gate fails (e.g. out of scope), citations list is reset to empty."""
    mock_retrieval_service = MagicMock()
    mock_retrieval_service.search = AsyncMock(
        return_value=SearchResponse(
            query="Out of scope medical question",
            results=[
                SearchResult(
                    text="Some text",
                    score=0.85,
                    percentage_score=85.0,
                    document_id="doc_chunk_1",
                    source="nice.pdf",
                    pdf_page=1,
                    document_page=1,
                    section="Sec 1",
                    recommendation_id=None,
                    is_table=False,
                )
            ]
        )
    )

    mock_llm_provider = MagicMock(spec=GeminiLLMProvider)
    # Gemini incorrectly returns citations even though is_in_scope is False
    mock_llm_provider.generate = AsyncMock(
        return_value=ClinicalLLMResponse(
            is_in_scope=False,  # Out of scope!
            is_knowledge_sufficient=True,
            answer=None,
            citations=[Citation(chunk_id="doc_chunk_1")],
            refusal_reason="Out of scope",
            provider="gemini",
            model_name="gemini-1.5-flash",
            filtered_chunks_count=1,
        )
    )

    service = GenerationService(
        retrieval_service=mock_retrieval_service,
        llm_provider=mock_llm_provider,
        min_score_threshold=80.0,
    )

    response = await service.generate_response("Out of scope medical question")

    # Assert AND gate cleared answer AND citations
    assert response.result.is_in_scope is False
    assert response.result.answer is None
    assert response.result.citations == []


def test_generation_route():
    """Test FastAPI generation route integration."""
    client = TestClient(app)

    response = client.post(
        "/api/v1/generation/generate",
        json={"query": "Diabetes management guidelines"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "result" in data
    assert "is_in_scope" in data["result"]
    assert "citations" in data["result"]
