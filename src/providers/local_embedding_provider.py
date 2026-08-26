import os
import gc
import logging
from typing import List, Optional
import torch

from interfaces.embedding_provider import EmbeddingProvider
from schemas.documents import EmbeddingResult
from config.settings import settings

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local In-Process Embedding Provider using BAAI/bge-m3 via FlagEmbedding.
    
    Generates unified dense (1024-dim) and sparse (lexical weights) embeddings directly in-process.
    Features:
      - Lazy loading: Model is only loaded on first inference call (won't download on import).
      - Device auto-detection: Uses CUDA if available with FP16, otherwise CPU.
      - Memory management: Clears PyTorch CUDA cache and runs garbage collection after batches.
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        expected_dimension: int = settings.EMBEDDING_DIMENSION,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
        device: str = settings.EMBEDDING_DEVICE,
    ):
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.device_setting = device
        self._model = None

    def _resolve_device(self) -> str:
        """Resolve device string based on configuration and hardware availability."""
        if self.device_setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device_setting

    @property
    def model(self):
        """Lazy load BGEM3FlagModel upon first access."""
        if self._model is None:
            resolved_device = self._resolve_device()
            use_fp16 = resolved_device.startswith("cuda") and torch.cuda.is_available()

            logger.info(
                f"🧠 Lazy loading local embedding model '{self.model_name}' "
                f"on device='{resolved_device}' (use_fp16={use_fp16})..."
            )

            try:
                from FlagEmbedding import BGEM3FlagModel
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=use_fp16,
                    device=resolved_device,
                )
                logger.info(f"✅ Local embedding model '{self.model_name}' loaded successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to load local embedding model '{self.model_name}': {e}", exc_info=True)
                raise RuntimeError(
                    f"Could not load local embedding model '{self.model_name}'. "
                    f"Ensure FlagEmbedding and PyTorch are installed. Error: {e}"
                )
        return self._model

    async def check_health(self) -> bool:
        """Health check for local embedding provider."""
        return True

    async def embed_documents(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate dense and sparse embeddings for a list of texts in batches."""
        if not texts:
            return []

        results: List[EmbeddingResult] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.info(f"🚀 Starting Local Embedding generation for {len(texts)} texts across {total_batches} batches...")

        for i in range(0, len(texts), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch_texts = texts[i : i + self.batch_size]
            logger.info(f"📦 Embedding Batch {batch_num}/{total_batches} ({min(i + self.batch_size, len(texts))}/{len(texts)} texts)")

            batch_results = self._encode_batch(batch_texts)
            results.extend(batch_results)

        logger.info(f"✅ All {len(texts)} texts embedded locally successfully!")
        return results

    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query text."""
        res = await self.embed_documents([text])
        if not res:
            raise RuntimeError("Failed to generate local embedding for query.")
        return res[0]

    def _encode_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """In-process encoding of a batch of texts using BGEM3FlagModel."""
        try:
            # Memory cleanup before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            out = self.model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=len(texts),
            )

            dense_vecs = out["dense_vecs"]
            lexical_weights = out["lexical_weights"]

            batch_results: List[EmbeddingResult] = []
            for d_vec, lex in zip(dense_vecs, lexical_weights):
                dense_list = d_vec.tolist() if hasattr(d_vec, "tolist") else list(d_vec)
                indices = [int(k) for k in lex.keys()]
                values = [float(v) for v in lex.values()]

                batch_results.append(
                    EmbeddingResult(
                        dense=dense_list,
                        sparse_indices=indices,
                        sparse_values=values,
                    )
                )

            # Memory cleanup after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return batch_results

        except Exception as e:
            logger.error(f"Error during local batch encoding: {e}", exc_info=True)
            raise RuntimeError(f"Local embedding inference failed: {e}")

