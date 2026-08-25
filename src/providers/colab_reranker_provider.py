import asyncio
import logging
from typing import List
import httpx

from interfaces.reranker import Reranker
from schemas.retrieval import RerankedDocument
from config.settings import settings

logger = logging.getLogger(__name__)


class ColabReranker(Reranker):
    """Reranker provider calling the remote Colab BGE Cross-Encoder service."""

    def __init__(
        self,
        api_url: str = settings.RERANKER_API_URL,
        api_key: str = settings.RERANKER_API_KEY,
        timeout: int = settings.RERANKER_TIMEOUT,
    ):
        self.api_url = api_url.rstrip("/") if api_url else ""
        self.api_key = api_key
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

    async def rerank(
        self, query: str, documents: List[RerankedDocument]
    ) -> List[RerankedDocument]:
        """Rerank candidate documents using remote cross-encoder API endpoint.
        
        Note: The API returns scores matching candidate order strictly by index.
        No sorting or mutating is done until scores are attached.
        """
        if not documents:
            return []

        if not self.api_url:
            raise ValueError("RERANKER_API_URL is not configured.")

        # Build request document list in fixed order
        doc_texts = [doc.text for doc in documents]
        payload = {"query": query, "documents": doc_texts}

        url = f"{self.api_url}/rerank"
        max_retries = 3

        logger.info(f"⚡ Sending {len(documents)} candidate chunks to Colab Reranker endpoint ({url})...")

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url, json=payload, headers=self._get_headers()
                    )

                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"Reranker API returned status HTTP {response.status_code}: {response.text}",
                        request=response.request,
                        response=response,
                    )

                data = response.json()
                if "scores" not in data or not isinstance(data["scores"], list):
                    raise ValueError(f"Malformed response from reranker API: missing 'scores' array. Got: {data}")

                scores = data["scores"]
                if len(scores) != len(documents):
                    raise ValueError(
                        f"Reranker returned {len(scores)} scores for {len(documents)} documents."
                    )

                # Map scores strictly by original document index
                reranked: List[RerankedDocument] = []
                for i, doc in enumerate(documents):
                    reranked.append(
                        RerankedDocument(
                            chunk_id=doc.chunk_id,
                            text=doc.text,
                            score=float(scores[i]),
                            metadata=doc.metadata,
                        )
                    )

                logger.info(f"✅ Colab Reranker successfully scored {len(reranked)} documents.")
                return reranked

            except (httpx.HTTPError, ValueError, KeyError) as e:
                logger.warning(f"Reranker attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Colab Reranker API failed after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Reranker request failed.")
