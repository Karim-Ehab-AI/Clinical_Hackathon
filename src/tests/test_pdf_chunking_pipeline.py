import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.pdf_chunking_pipeline import PDFChunkingPipeline
from schemas.documents import ChunkMetadata, DocumentChunk


def test_content_role_derivation():
    """Test rule-based content_role mapping against heading levels."""
    pipeline = PDFChunkingPipeline()

    assert pipeline._derive_content_role(["First Aid Management", "Immediate Action Steps"]) == "key_action"
    assert pipeline._derive_content_role(["Chapter 1", "Introduction & Scope"]) == "introduction"
    assert pipeline._derive_content_role(["General Care", "Clinical Best Practice Points"]) == "good_practice"
    assert pipeline._derive_content_role(["Burns", "First Aid Steps and Treatment"]) == "first_aid_steps"
    assert pipeline._derive_content_role(["Medication", "Warnings and Contraindications"]) == "caution"
    assert pipeline._derive_content_role(["Emergency", "When to Call Emergency Help"]) == "access_help"
    assert pipeline._derive_content_role(["Post-treatment", "Patient Recovery and Aftercare"]) == "recovery"
    assert pipeline._derive_content_role(["Public Health", "Community Prevention Education"]) == "education"
    assert pipeline._derive_content_role(["Guidelines", "Evidence and Scientific Rationale"]) == "scientific_foundation"
    assert pipeline._derive_content_role(["Chapter 5", "Miscellaneous Topic"]) is None


def test_content_type_mapping():
    """Test doc_item_labels mapping to content_type."""
    pipeline = PDFChunkingPipeline()

    assert pipeline._map_content_type(["table", "table_body"]) == "table"
    assert pipeline._map_content_type(["picture", "caption"]) == "figure"
    assert pipeline._map_content_type(["paragraph", "text"]) == "text"
    assert pipeline._map_content_type(["table", "picture"]) == "text"  # Mixed label fallback


def test_token_count_local():
    """Test token count using BGE-M3 local tokenizer."""
    pipeline = PDFChunkingPipeline()

    text = "First aid treatment for burns involves applying cold water."
    count = pipeline._calculate_token_count(text)
    assert count > 0


@pytest.mark.asyncio
async def test_full_remote_chunking_contract_mock(tmp_path):
    """Test full remote chunking & embedding pipeline against mock API responses."""
    pipeline = PDFChunkingPipeline()

    # Create dummy pdf file
    dummy_pdf = tmp_path / "test_guideline.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 Mock PDF Content For Testing Pipeline")

    mock_chunk_pdf_response = {
        "total_pages": 1,
        "chunk_count": 1,
        "chunks": [
            {
                "chunk_index": 0,
                "text": "Apply cold water to the burn immediately.",
                "contextualized_text": "First Aid -> Burns -> Apply cold water to the burn immediately.",
                "headings": ["First Aid", "Burns"],
                "pages": [1],
                "doc_item_labels": ["text"]
            }
        ]
    }

    mock_embed_response = {
        "dense": [[0.1, 0.2, 0.3]],
        "sparse": [
            {
                "indices": [10, 20],
                "values": [0.5, 0.8]
            }
        ],
        "dense_size": 1024
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        res1 = MagicMock()
        res1.status_code = 200
        res1.json.return_value = mock_chunk_pdf_response

        res2 = MagicMock()
        res2.status_code = 200
        res2.json.return_value = mock_embed_response

        mock_post.side_effect = [res1, res2]

        final_chunks = await pipeline.process_pdf_remote(
            pdf_path=str(dummy_pdf),
            remote_base_url="http://mock-remote:8000"
        )

        assert len(final_chunks) == 1
        chunk = final_chunks[0]
        assert chunk.text == "Apply cold water to the burn immediately."
        assert chunk.metadata.content_role == "first_aid_steps"
        assert chunk.metadata.section == "First Aid"
        assert chunk.metadata.subsection == "Burns"
        assert chunk.dense_vector == [0.1, 0.2, 0.3]
        assert chunk.sparse_indices == [10, 20]
        assert chunk.sparse_values == [0.5, 0.8]
