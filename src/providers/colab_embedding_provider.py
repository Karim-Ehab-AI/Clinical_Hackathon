import asyncio
import logging
from typing import List
import httpx

from interfaces.embedding_provider import EmbeddingProvider
from schemas.documents import EmbeddingResult
from config.settings import settings

logger = logging.getLogger(__name__)


class ColabEmbeddingProvider(EmbeddingProvider):
    """Embedding Provider calling the remote Colab BGE-M3 HTTP API service.
    Returns unified dense (1024-dim) and sparse (lexical weight) embeddings per text batch.
    """

    def __init__(
        self,
        api_url: str = settings.EMBEDDING_API_URL,
        api_key: str = settings.EMBEDDING_API_KEY,
        expected_dimension: int = settings.EMBEDDING_DIMENSION,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
        timeout: int = settings.EMBEDDING_TIMEOUT,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.timeout = timeout

    def _get_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "ClinicalRAG-Client/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


    async def check_health(self) -> bool:
        """Query remote Colab endpoint GET /health to verify status."""
        url = f"{self.api_url}/health"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.get(url, headers=self._get_headers())
                if res.status_code == 200:
                    data = res.json()
                    logger.info(f"Colab embedding healthcheck OK: {data}")
                    return True
        except Exception as e:
            logger.warning(f"Colab embedding healthcheck failed: {e}")
        return False

    async def embed_documents(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a list of text documents in batches with progress logging."""
        if not texts:
            return []

        results: List[EmbeddingResult] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(f"🚀 Starting Embedding generation for {len(texts)} chunks across {total_batches} batches...")

        # Process in batches of size batch_size
        for i in range(0, len(texts), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            logger.info(f"📦 Embedding Batch {batch_num}/{total_batches} ({min(i + self.batch_size, len(texts))}/{len(texts)} chunks)")
            batch_texts = texts[i : i + self.batch_size]
            batch_results = await self._embed_batch_with_retry(batch_texts)
            results.extend(batch_results)

        logger.info(f"✅ All {len(texts)} chunks embedded successfully!")
        return results


    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query text."""
        res = await self.embed_documents([text])
        if not res:
            raise RuntimeError("Failed to generate embedding for query.")
        return res[0]

    async def _embed_batch_with_retry(self, texts: List[str], max_retries: int = 3) -> List[EmbeddingResult]:
        """Call POST /embed with retry and backoff logic."""
        url = f"{self.api_url}/embed"
        payload = {"texts": texts}

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=self._get_headers())

                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"Embedding API returned HTTP {response.status_code}: {response.text}",
                        request=response.request,
                        response=response,
                    )

                data = response.json()

                # Validate response structure
                if "dense" not in data or "sparse" not in data:
                    raise ValueError(f"Malformed response from embedding API: missing dense/sparse keys. Got: {list(data.keys())}")

                # Validate dimension
                returned_dim = data.get("dense_size", len(data["dense"][0]) if data["dense"] else 0)
                if returned_dim != self.expected_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.expected_dimension}, got {returned_dim} from endpoint."
                    )

                dense_list = data["dense"]
                sparse_list = data["sparse"]

                if len(dense_list) != len(texts) or len(sparse_list) != len(texts):
                    raise ValueError("Embedding response count does not match input texts count.")

                batch_results = []
                for d_vec, s_dict in zip(dense_list, sparse_list):
                    batch_results.append(
                        EmbeddingResult(
                            dense=d_vec,
                            sparse_indices=s_dict.get("indices", []),
                            sparse_values=s_dict.get("values", []),
                        )
                    )

                return batch_results

            except (httpx.HTTPError, ValueError, KeyError) as e:
                logger.warning(f"Embedding attempt {attempt}/{max_retries} failed for {url}: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Embedding API unavailable or failed after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Embedding API request failed.")
