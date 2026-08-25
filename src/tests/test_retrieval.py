import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from fastapi.testclient import TestClient

from main import app
from schemas.documents import EmbeddingResult
from schemas.retrieval import RerankedDocument
from services.retrieval_service import RetrievalService
from services.reranking_service import RerankingService
from providers.colab_reranker_provider import ColabReranker
from providers.noop_reranker_provider import NoOpReranker
from providers.qdrant_provider import QdrantProvider


# ---------------------------------------------------------------------------
# 1. RRF Fusion Logic Tests
# ---------------------------------------------------------------------------
def test_reciprocal_rank_fusion_ordering():
    service = RetrievalService(
        embedding_provider=MagicMock(),
        vector_store=MagicMock(),
        reranking_service=MagicMock(),
    )

    dense_hits = [
        {"id": "doc1", "payload": {"chunk_id": "doc1", "text": "Doc 1 text"}},
        {"id": "doc2", "payload": {"chunk_id": "doc2", "text": "Doc 2 text"}},
    ]
    sparse_hits = [
        {"id": "doc2", "payload": {"chunk_id": "doc2", "text": "Doc 2 text"}},
        {"id": "doc3", "payload": {"chunk_id": "doc3", "text": "Doc 3 text"}},
    ]

    # k=60
    # doc1: dense rank 1 -> 1/(60+1) = 1/61 = 0.016393
    # doc2: dense rank 2 -> 1/(60+2) = 1/62, sparse rank 1 -> 1/(60+1) = 1/61. Total = 1/62 + 1/61 = 0.032523
    # doc3: sparse rank 2 -> 1/(60+2) = 1/62 = 0.016129

    fused = service.reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_k=10)

    assert len(fused) == 3
    assert fused[0].chunk_id == "doc2"  # Highest combined RRF score
    assert fused[1].chunk_id == "doc1"
    assert fused[2].chunk_id == "doc3"
    assert fused[0].score > fused[1].score > fused[2].score


# ---------------------------------------------------------------------------
# 2. Qdrant Hybrid Search Mock Tests
# ---------------------------------------------------------------------------
def test_qdrant_hybrid_search():
    provider = QdrantProvider(url="http://mock-qdrant:6333", collection_name="test_col")
    provider._client = MagicMock()

    mock_dense_point = MagicMock()
    mock_dense_point.id = "p1"
    mock_dense_point.score = 0.95
    mock_dense_point.payload = {"text": "dense match", "chunk_id": "p1"}

    mock_sparse_point = MagicMock()
    mock_sparse_point.id = "p2"
    mock_sparse_point.score = 0.85
    mock_sparse_point.payload = {"text": "sparse match", "chunk_id": "p2"}

    provider._client.query_points.side_effect = [
        MagicMock(points=[mock_dense_point]),
        MagicMock(points=[mock_sparse_point]),
    ]
    provider._client.get_collections.return_value = MagicMock(
        collections=[MagicMock(name="test_col")]
    )

    dense_hits, sparse_hits = provider.hybrid_search(
        dense_vector=[0.1] * 1024,
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.8],
        dense_top_k=5,
        sparse_top_k=5,
    )

    assert len(dense_hits) == 1
    assert dense_hits[0]["id"] == "p1"
    assert len(sparse_hits) == 1
    assert sparse_hits[0]["id"] == "p2"
    assert provider._client.query_points.call_count == 2


# ---------------------------------------------------------------------------
# 3. ColabReranker Provider Mock Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_colab_reranker_success():
    reranker = ColabReranker(api_url="http://mock-reranker:8000")
    docs = [
        RerankedDocument(chunk_id="c1", text="Chunk 1", score=0.01, metadata={}),
        RerankedDocument(chunk_id="c2", text="Chunk 2", score=0.02, metadata={}),
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"scores": [0.95, 0.40]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        reranked = await reranker.rerank("diabetes target", docs)

        assert len(reranked) == 2
        # Scores attached strictly by index
        assert reranked[0].chunk_id == "c1"
        assert reranked[0].score == 0.95
        assert reranked[1].chunk_id == "c2"
        assert reranked[1].score == 0.40


# ---------------------------------------------------------------------------
# 4. NoOpReranker & Fallback Behavior Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reranking_service_fallback_on_error():
    failing_reranker = MagicMock()
    failing_reranker.rerank = AsyncMock(side_effect=RuntimeError("Endpoint down"))

    service = RerankingService(reranker=failing_reranker)

    docs = [
        RerankedDocument(chunk_id="c1", text="Doc 1", score=0.05, metadata={}),
        RerankedDocument(chunk_id="c2", text="Doc 2", score=0.03, metadata={}),
    ]

    # Should not raise exception, but fallback to NoOp maintaining order & fused scores
    result = await service.rerank("query", docs)
    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].score == 0.05
    assert result[1].chunk_id == "c2"
    assert result[1].score == 0.03


@pytest.mark.asyncio
async def test_reranking_service_explicit_sort():
    mock_reranker = MagicMock()
    # Mock reranker returning items out of order by score
    mock_reranker.rerank = AsyncMock(
        return_value=[
            RerankedDocument(chunk_id="c1", text="Doc 1", score=0.30, metadata={}),
            RerankedDocument(chunk_id="c2", text="Doc 2", score=0.90, metadata={}),
        ]
    )

    service = RerankingService(reranker=mock_reranker)
    docs = [
        RerankedDocument(chunk_id="c1", text="Doc 1", score=0.05, metadata={}),
        RerankedDocument(chunk_id="c2", text="Doc 2", score=0.03, metadata={}),
    ]

    result = await service.rerank("query", docs)
    # Must be explicitly sorted descending by score: c2 (0.90) first, c1 (0.30) second
    assert result[0].chunk_id == "c2"
    assert result[0].score == 0.90
    assert result[1].chunk_id == "c1"
    assert result[1].score == 0.30


# ---------------------------------------------------------------------------
# 5. Retrieval Route End-to-End Test
# ---------------------------------------------------------------------------
def test_retrieval_search_route():
    client = TestClient(app)

    mock_emb_provider = AsyncMock()
    mock_emb_provider.embed_query.return_value = EmbeddingResult(
        dense=[0.1] * 1024, sparse_indices=[10], sparse_values=[0.9]
    )

    mock_vector_store = MagicMock()
    mock_vector_store.hybrid_search.return_value = (
        [
            {
                "id": "p1",
                "score": 0.88,
                "payload": {
                    "chunk_id": "chunk_101",
                    "text": "HbA1c target for type 2 diabetes is < 48 mmol/mol.",
                    "document_id": "doc_abc",
                    "source": "NICE_NG28.pdf",
                    "pdf_page": 13,
                    "document_page": 12,
                    "section": "1.5 Targets",
                    "recommendation_id": "1.5.7",
                    "is_table": False,
                },
            }
        ],
        [],
    )

    mock_reranking_service = AsyncMock()
    mock_reranking_service.rerank.return_value = [
        RerankedDocument(
            chunk_id="chunk_101",
            text="HbA1c target for type 2 diabetes is < 48 mmol/mol.",
            score=0.98,
            metadata={
                "document_id": "doc_abc",
                "source": "NICE_NG28.pdf",
                "pdf_page": 13,
                "document_page": 12,
                "section": "1.5 Targets",
                "recommendation_id": "1.5.7",
                "is_table": False,
            },
        )
    ]

    mock_service = RetrievalService(
        embedding_provider=mock_emb_provider,
        vector_store=mock_vector_store,
        reranking_service=mock_reranking_service,
    )

    with patch("controllers.retrieval_controller.RetrievalService", return_value=mock_service):
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "What HbA1c target is recommended for adults with type 2 diabetes?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What HbA1c target is recommended for adults with type 2 diabetes?"
        assert len(data["results"]) == 1
        res = data["results"][0]
        assert res["score"] == 0.98
        assert res["percentage_score"] == 98.0
        assert res["recommendation_id"] == "1.5.7"
        assert res["pdf_page"] == 13
        assert res["document_page"] == 12
        assert res["section"] == "1.5 Targets"
        assert res["is_table"] is False
