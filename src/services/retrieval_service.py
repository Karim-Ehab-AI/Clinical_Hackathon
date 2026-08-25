import logging
from typing import List, Dict, Optional

from interfaces.embedding_provider import EmbeddingProvider
from interfaces.vector_store import VectorStore
from providers.colab_embedding_provider import ColabEmbeddingProvider
from providers.qdrant_provider import QdrantProvider
from services.reranking_service import RerankingService
from schemas.retrieval import RerankedDocument, SearchResult, SearchResponse, RerankResponse
from config.settings import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service orchestrating dense+sparse query embedding, Qdrant search, RRF fusion, and reranking."""

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        reranking_service: Optional[RerankingService] = None,
    ):
        self.embedding_provider = embedding_provider or ColabEmbeddingProvider()
        self.vector_store = vector_store or QdrantProvider()
        self.reranking_service = reranking_service or RerankingService()

    def reciprocal_rank_fusion(
        self,
        dense_hits: List[dict],
        sparse_hits: List[dict],
        k: int = settings.RRF_K,
        top_k: int = settings.HYBRID_TOP_K,
    ) -> List[RerankedDocument]:
        """Combine dense and sparse candidate search results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, dict] = {}

        def process_hits(hits: List[dict]):
            for rank_idx, hit in enumerate(hits):
                payload = hit.get("payload", {})
                chunk_id = payload.get("chunk_id") or hit.get("id")
                if not chunk_id:
                    continue

                rank = rank_idx + 1
                score_contrib = 1.0 / (k + rank)

                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    doc_map[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": payload.get("text", ""),
                        "metadata": payload,
                    }
                rrf_scores[chunk_id] += score_contrib

        process_hits(dense_hits)
        process_hits(sparse_hits)

        fused_docs: List[RerankedDocument] = []
        for chunk_id, score in rrf_scores.items():
            info = doc_map[chunk_id]
            fused_docs.append(
                RerankedDocument(
                    chunk_id=chunk_id,
                    text=info["text"],
                    score=score,
                    metadata=info["metadata"],
                )
            )

        # Sort descending by fused RRF score
        fused_docs.sort(key=lambda d: d.score, reverse=True)
        logger.info(f"RRF Fusion completed: merged {len(dense_hits)} dense + {len(sparse_hits)} sparse hits into {len(fused_docs)} unique docs (top {top_k} retained).")
        return fused_docs[:top_k]

    async def search(
        self,
        query: str,
        dense_top_k: int = settings.DENSE_TOP_K,
        sparse_top_k: int = settings.SPARSE_TOP_K,
        hybrid_top_k: int = settings.HYBRID_TOP_K,
        rerank_top_k: int = settings.RERANK_TOP_K,
    ) -> SearchResponse:
        """Execute end-to-end retrieval flow: embed -> search -> fuse -> rerank."""
        clean_query = query.strip()
        logger.info(f"🔍 Retrieval started for query (length: {len(clean_query)} chars)")

        # 1. Embed query (dense + sparse)
        query_embedding = await self.embedding_provider.embed_query(clean_query)

        # 2. Hybrid vector search in Qdrant
        dense_hits, sparse_hits = self.vector_store.hybrid_search(
            dense_vector=query_embedding.dense,
            sparse_indices=query_embedding.sparse_indices,
            sparse_values=query_embedding.sparse_values,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
        )
        logger.info(f"Qdrant retrieval completed: {len(dense_hits)} dense candidates, {len(sparse_hits)} sparse candidates.")

        # 3. RRF Fusion
        fused_candidates = self.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            k=settings.RRF_K,
            top_k=hybrid_top_k,
        )

        # 4. Reranking (with fallback handling inside RerankingService)
        reranked_docs = await self.reranking_service.rerank(clean_query, fused_candidates)

        # 5. Take top RERANK_TOP_K ranked candidates
        final_docs = reranked_docs[:rerank_top_k]
        logger.info(f"Retrieval complete: returning top {len(final_docs)} results.")

        # 6. Map to response schema with score normalization
        max_rrf_score = 2.0 / (settings.RRF_K + 1)
        search_results: List[SearchResult] = []
        for doc in final_docs:
            meta = doc.metadata
            if doc.score <= 0.1:
                norm_score = min(1.0, doc.score / max_rrf_score)
            else:
                norm_score = min(1.0, max(0.0, doc.score))

            pct_score = round(norm_score * 100.0, 2)

            pdf_pages_list = meta.get("pdf_pages") or []
            actual_page = meta.get("pdf_page")
            if (actual_page is None or actual_page == 1) and isinstance(pdf_pages_list, list) and len(pdf_pages_list) > 0:
                actual_page = pdf_pages_list[0]
            if actual_page is None:
                actual_page = 1

            search_results.append(
                SearchResult(
                    text=doc.text,
                    score=round(norm_score, 4),
                    percentage_score=pct_score,
                    document_id=meta.get("document_id", ""),
                    source=meta.get("source", ""),
                    pdf_page=int(actual_page),
                    document_page=int(actual_page),
                    section=meta.get("section", ""),
                    recommendation_id=meta.get("recommendation_id"),
                    is_table=meta.get("is_table", False),
                )
            )

        return SearchResponse(query=clean_query, results=search_results)

    async def rerank_standalone(self, query: str, documents: List[str]) -> RerankResponse:
        """Standalone reranking endpoint allowing direct evaluation of custom candidate texts."""
        candidates = [
            RerankedDocument(chunk_id=f"doc_{i}", text=doc_text, score=0.0, metadata={})
            for i, doc_text in enumerate(documents)
        ]
        reranked = await self.reranking_service.rerank(query.strip(), candidates)
        scores = [round(doc.score, 4) for doc in reranked]
        results = [
            {
                "rank": i + 1,
                "text": doc.text,
                "score": round(doc.score, 4),
                "percentage_score": round(doc.score * 100.0, 2) if doc.score > 0.1 else round(min(1.0, doc.score / (2.0 / (settings.RRF_K + 1))) * 100.0, 2),
            }
            for i, doc in enumerate(reranked)
        ]
        return RerankResponse(query=query, scores=scores, results=results)


