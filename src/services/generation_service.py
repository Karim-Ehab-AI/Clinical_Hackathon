import logging
from typing import Optional, List

from services.retrieval_service import RetrievalService
from providers.gemini_llm_provider import GeminiLLMProvider
from schemas.llm import GenerateResponse, ClinicalLLMResponse
from config.settings import settings

logger = logging.getLogger(__name__)


class GenerationService:
    """End-to-End Clinical Generation Service with Score Threshold Filtering (Top 3 >= 80%) & Native Citation Tracking."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm_provider: Optional[GeminiLLMProvider] = None,
        min_score_threshold: float = settings.MIN_SIMILARITY_SCORE_THRESHOLD,
    ):
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_provider = llm_provider or GeminiLLMProvider()
        self.min_score_threshold = min_score_threshold

    async def generate_response(self, query: str) -> GenerateResponse:
        """Execute clinical RAG generation pipeline: Retrieval -> Filter Top 3 (>=80%) -> Gemini LLM + Citations -> AND Gate."""
        clean_query = query.strip()
        logger.info(f"🚀 Starting Clinical Generation Pipeline for query: '{clean_query[:60]}...'")

        # 1. Retrieve candidates via Hybrid Vector Search + RRF Fusion
        retrieval_response = await self.retrieval_service.search(query=clean_query)
        raw_results = retrieval_response.results

        # 2. Filter retrieved candidates by similarity score threshold (>= 80%)
        filtered_candidates: List[dict] = []
        for idx, item in enumerate(raw_results, 1):
            if item.percentage_score >= self.min_score_threshold:
                # Generate guaranteed unique chunk_id per candidate chunk
                raw_chunk_id = getattr(item, "chunk_id", None)
                if not raw_chunk_id or raw_chunk_id == item.document_id:
                    unique_chunk_id = f"chunk_p{item.pdf_page}_{idx}"
                else:
                    unique_chunk_id = str(raw_chunk_id)

                filtered_candidates.append({
                    "chunk_id": unique_chunk_id,
                    "document_id": item.document_id,
                    "source": item.source,
                    "text": item.text,
                    "score": item.score,
                    "percentage_score": item.percentage_score,
                    "pdf_page": item.pdf_page,
                    "section": item.section,
                    "recommendation_id": item.recommendation_id,
                    "is_table": item.is_table,
                })

        retrieved_count = len(raw_results)
        total_filtered_count = len(filtered_candidates)

        # Select ONLY top 3 candidates matching threshold
        top_3_candidates = filtered_candidates[:3]
        selected_count = len(top_3_candidates)

        logger.info(
            f"📊 Retrieval Stats: {retrieved_count} total retrieved | "
            f"{total_filtered_count} passed >= {self.min_score_threshold}% threshold | "
            f"{selected_count} top chunks passed to Gemini."
        )

        # 3. Guardrail Layer 1: If 0 candidates pass >=80% threshold, skip LLM call entirely
        if selected_count == 0:
            logger.warning(f"⚠️ 0 chunks passed {self.min_score_threshold}% threshold. Refusing generation without calling LLM.")
            
            # Match query language for refusal message
            is_arabic_query = any('\u0600' <= char <= '\u06FF' for char in clean_query)
            refusal_text = (
                "عذراً، المعلومات الطبية المتوفرة في المنظومة غير كافية لتقديم إجابة موثوقة (نسبة الثقة في نتائج البحث أقل من 80%)."
                if is_arabic_query else
                "Apologies, the available clinical evidence in the system is insufficient to provide a reliable answer (confidence score below 80%)."
            )

            refusal_result = ClinicalLLMResponse(
                is_in_scope=True,
                is_knowledge_sufficient=False,
                answer=None,
                citations=[],
                refusal_reason=refusal_text,
                provider="gemini",
                model_name=settings.GEMINI_MODEL,
                filtered_chunks_count=0,
            )
            return GenerateResponse(
                query=clean_query,
                result=refusal_result,
                retrieved_chunks_count=retrieved_count,
                filtered_chunks_count=0,
            )

        # 4. Invoke Gemini Provider with top 3 filtered chunks
        llm_response = await self.llm_provider.generate(
            query=clean_query,
            filtered_docs=top_3_candidates,
        )

        # 5. Guardrail Layer 2: Local AND-Gate verification
        if not (llm_response.is_in_scope and llm_response.is_knowledge_sufficient):
            logger.info("🛡️ AND-Gate Refusal triggered by LLM evaluation.")
            llm_response.answer = None
            llm_response.citations = []

        return GenerateResponse(
            query=clean_query,
            result=llm_response,
            retrieved_chunks_count=retrieved_count,
            filtered_chunks_count=selected_count,
        )
