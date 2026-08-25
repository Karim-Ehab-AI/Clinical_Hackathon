from unittest.mock import MagicMock
from providers.qdrant_provider import QdrantProvider
from schemas.documents import DocumentChunk, ChunkMetadata, EmbeddingResult


def test_qdrant_provider_point_formatting(monkeypatch):
    mock_client = MagicMock()

    provider = QdrantProvider(url="http://localhost:6333", collection_name="clinical_documents")
    provider._client = mock_client

    chunk = DocumentChunk(
        chunk_id="chunk-123",
        text="Sample clinical text content",
        metadata=ChunkMetadata(
            chunk_id="chunk-123",
            document_id="hash_abc",
            document_title="Sample Title",
            source="src/assets/hash_abc.pdf",
            pdf_page=1,
            document_page=1,
            section="Section 1",
            recommendation_id="1.5.7",
            recommendation_class="IIa",
            evidence_level="B",
        ),
    )

    embedding = EmbeddingResult(
        dense=[0.1] * 1024,
        sparse_indices=[1, 5, 10],
        sparse_values=[0.2, 0.4, 0.6],
    )

    count = provider.upsert_points([chunk], [embedding])

    assert count == 1
    assert mock_client.upsert.called
    call_args = mock_client.upsert.call_args[1]
    points = call_args["points"]
    assert len(points) == 1
    point = points[0]

    # Verify named vector formatting
    assert "dense" in point.vector
    assert len(point.vector["dense"]) == 1024
    assert "sparse" in point.vector
    assert point.vector["sparse"].indices == [1, 5, 10]
    assert point.vector["sparse"].values == [0.2, 0.4, 0.6]

    # Verify payload contents
    payload = point.payload
    assert payload["document_id"] == "hash_abc"
    assert payload["recommendation_id"] == "1.5.7"
    assert payload["recommendation_class"] == "IIa"
    assert payload["evidence_level"] == "B"
    assert payload["text"] == "Sample clinical text content"
