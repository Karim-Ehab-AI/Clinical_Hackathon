import pytest
import httpx
from providers.colab_embedding_provider import ColabEmbeddingProvider


@pytest.mark.asyncio
async def test_colab_embedding_provider_response_parsing(monkeypatch):
    async def mock_post(*args, **kwargs):
        return httpx.Response(
            status_code=200,
            json={
                "dense": [[0.1] * 1024, [0.2] * 1024],
                "sparse": [
                    {"indices": [10, 20], "values": [0.5, 0.8]},
                    {"indices": [30, 40], "values": [0.1, 0.9]},
                ],
                "dense_size": 1024,
            },
            request=httpx.Request("POST", "http://localhost:8000/embed"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = ColabEmbeddingProvider(api_url="http://localhost:8000", expected_dimension=1024)
    results = await provider.embed_documents(["Text 1", "Text 2"])

    assert len(results) == 2
    assert len(results[0].dense) == 1024
    assert results[0].sparse_indices == [10, 20]
    assert results[0].sparse_values == [0.5, 0.8]


@pytest.mark.asyncio
async def test_colab_embedding_provider_dimension_mismatch(monkeypatch):
    async def mock_post(*args, **kwargs):
        return httpx.Response(
            status_code=200,
            json={
                "dense": [[0.1] * 768],
                "sparse": [{"indices": [10], "values": [0.5]}],
                "dense_size": 768,
            },
            request=httpx.Request("POST", "http://localhost:8000/embed"),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    provider = ColabEmbeddingProvider(api_url="http://localhost:8000", expected_dimension=1024, batch_size=32)

    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        await provider.embed_documents(["Sample text"])
