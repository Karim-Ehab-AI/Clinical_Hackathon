import json
import logging
from typing import List, Optional
import httpx

from interfaces.llm_provider import LLMProvider
from schemas.llm import ClinicalLLMResponse, Citation
from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are an expert First Aid Clinical Assistant specializing strictly in First Aid, Emergency Procedures, and Initial Medical Response.

CRITICAL EVALUATION & FORMATTING RULES:

1. DOMAIN SCOPE CHECK (`is_in_scope`):
   - Scope is STRICTLY First Aid, emergency procedures, triage, basic life support, trauma response, CPR, burns, choking, bleeding, fractures, poisoning, and initial emergency guidelines.
   - If the query asks about unrelated non-emergency clinical topics, complex long-term therapies, or non-medical topics, set `is_in_scope = false`.

2. KNOWLEDGE SUFFICIENCY CHECK (`is_knowledge_sufficient`):
   - Evaluate if the provided context chunks contain sufficient first aid evidence to provide safe guidance.
   - If context is empty or lacks specific guidance for the emergency query, set `is_knowledge_sufficient = false`.

3. MANDATORY AND-GATE & CITATIONS RULE:
   - If `is_in_scope` is false OR `is_knowledge_sufficient` is false:
     - Set `answer = null`.
     - Set `citations = []`.
     - Provide a polite, professional `refusal_reason` explaining why in the user's language.
   - ONLY when BOTH `is_in_scope` is true AND `is_knowledge_sufficient` is true:
     - Set `answer` to a concise, actionable first-aid response.
     - Set `refusal_reason = null`.
     - Include in `citations` ONLY the chunks actually drawn upon to construct the answer.

4. FORMATTING RULES (BULLET POINTS & CONCISENESS):
   - Format all emergency instructions, directions, and action steps in clear BULLET POINTS.
   - Keep the answer as CONCISE and DIRECT as possible. Avoid unnecessary fluff or long text.

5. LANGUAGE MATCHING RULE:
   - Always respond in the EXACT SAME language as the user's query!
   - If the user asks in Arabic, write `answer` and `refusal_reason` strictly in Arabic.
   - If the user asks in English, write `answer` and `refusal_reason` strictly in English.
"""


class GeminiLLMProvider(LLMProvider):
    """Google Gemini API Provider for Clinical LLM Generation with Citation Tracking."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        self.timeout = timeout
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def _format_context(self, docs: List[dict]) -> str:
        if not docs:
            return "No clinical context available."

        formatted_chunks = []
        for idx, doc in enumerate(docs, 1):
            chunk_id = doc.get("chunk_id", f"chunk_{idx}")
            text = doc.get("text", "")
            pdf_page = doc.get("pdf_page", 1)
            section = doc.get("section", "")
            rec_id = doc.get("recommendation_id") or "N/A"
            score = doc.get("percentage_score", 0.0)

            source = doc.get("source", "")
            chunk_str = (
                f"--- CHUNK HEADER ---\n"
                f"chunk_id: {chunk_id}\n"
                f"source_file: {source}\n"
                f"recommendation_id: {rec_id}\n"
                f"pdf_page: {pdf_page}\n"
                f"section: {section}\n"
                f"confidence_score: {score}%\n"
                f"--- TEXT CONTENT ---\n{text}\n"
            )
            formatted_chunks.append(chunk_str)

        return "\n\n".join(formatted_chunks)

    async def generate(
        self,
        query: str,
        filtered_docs: List[dict],
        system_prompt: Optional[str] = None,
    ) -> ClinicalLLMResponse:
        """Generate structured clinical response using Google Gemini API with Citation tracking."""
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. Returning default fallback refusal.")
            return ClinicalLLMResponse(
                is_in_scope=True,
                is_knowledge_sufficient=False,
                answer=None,
                citations=[],
                refusal_reason="مفتاح Gemini API غير متاح في الإعدادات (.env). يرجى تعيين GEMINI_API_KEY.",
                provider="gemini",
                model_name=self.model_name,
                filtered_chunks_count=len(filtered_docs),
            )

        context_text = self._format_context(filtered_docs)
        sys_instructions = system_prompt or DEFAULT_SYSTEM_PROMPT

        user_content = f"CLINICAL CONTEXT CHUNKS:\n{context_text}\n\nUSER CLINICAL QUERY: {query}"

        # Define JSON Response Schema for Gemini Structured Outputs
        json_schema = {
            "type": "OBJECT",
            "properties": {
                "is_in_scope": {
                    "type": "BOOLEAN",
                    "description": "True if query is strictly about First Aid / Emergency Response. False otherwise."
                },
                "is_knowledge_sufficient": {
                    "type": "BOOLEAN",
                    "description": "True if context contains sufficient first aid clinical evidence. False otherwise."
                },
                "answer": {
                    "type": "STRING",
                    "nullable": True,
                    "description": "Concise first-aid response formatted in bullet points. Must be null if either flag is false."
                },
                "citations": {
                    "type": "ARRAY",
                    "description": "List of chunks actually used to construct the answer. Must be empty if answer is null.",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "chunk_id": {"type": "STRING"},
                            "recommendation_id": {"type": "STRING", "nullable": True},
                            "pdf_page": {"type": "INTEGER", "nullable": True},
                            "section": {"type": "STRING", "nullable": True}
                        },
                        "required": ["chunk_id"]
                    }
                },
                "refusal_reason": {
                    "type": "STRING",
                    "nullable": True,
                    "description": "Polite refusal message in user's query language if out of scope or insufficient knowledge."
                }
            },
            "required": ["is_in_scope", "is_knowledge_sufficient"]
        }

        payload = {
            "contents": [
                {
                    "parts": [{"text": user_content}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": sys_instructions}]
            },
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": json_schema
            }
        }

        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}?key={self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"⚡ Sending generation request to Gemini ({self.model_name})...")
                res = await client.post(url, json=payload, headers=headers)

                if res.status_code != 200:
                    logger.error(f"Gemini API error ({res.status_code}): {res.text}")
                    return ClinicalLLMResponse(
                        is_in_scope=True,
                        is_knowledge_sufficient=False,
                        answer=None,
                        citations=[],
                        refusal_reason=f"خطأ من Gemini API: {res.status_code}",
                        provider="gemini",
                        model_name=self.model_name,
                        filtered_chunks_count=len(filtered_docs),
                    )

                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return ClinicalLLMResponse(
                        is_in_scope=True,
                        is_knowledge_sufficient=False,
                        answer=None,
                        citations=[],
                        refusal_reason="لم يتم التوصل لاستجابة من موديل Gemini.",
                        provider="gemini",
                        model_name=self.model_name,
                        filtered_chunks_count=len(filtered_docs),
                    )

                raw_json_str = candidates[0]["content"]["parts"][0]["text"]
                parsed_data = json.loads(raw_json_str)

                is_in_scope = bool(parsed_data.get("is_in_scope", False))
                is_knowledge_sufficient = bool(parsed_data.get("is_knowledge_sufficient", False))
                answer = parsed_data.get("answer")
                refusal_reason = parsed_data.get("refusal_reason")

                # Build lookup dict for original doc text
                docs_by_id = {str(d.get("chunk_id")): d for d in filtered_docs}

                raw_citations = parsed_data.get("citations", [])
                citations_list: List[Citation] = []
                if isinstance(raw_citations, list):
                    for c in raw_citations:
                        if isinstance(c, dict) and "chunk_id" in c:
                            cid = str(c["chunk_id"])
                            matched_doc = docs_by_id.get(cid, {})
                            matched_text = matched_doc.get("text", "")

                            citations_list.append(
                                Citation(
                                    chunk_id=cid,
                                    source=matched_doc.get("source"),
                                    document_id=matched_doc.get("document_id"),
                                    recommendation_id=c.get("recommendation_id") if c.get("recommendation_id") and str(c.get("recommendation_id")).upper() != "N/A" else matched_doc.get("recommendation_id"),
                                    pdf_page=c.get("pdf_page") or matched_doc.get("pdf_page"),
                                    section=c.get("section") if c.get("section") and str(c.get("section")).upper() != "N/A" else matched_doc.get("section"),
                                    source_text=matched_text,
                                    score=matched_doc.get("score"),
                                    percentage_score=matched_doc.get("percentage_score"),
                                )
                            )

                # Enforce AND-Gate locally as additional guardrail
                if not (is_in_scope and is_knowledge_sufficient):
                    answer = None
                    citations_list = []
                    if not refusal_reason:
                        if not is_in_scope:
                            refusal_reason = "عذراً، هذا السؤال خارج اختصاص المنظومة الطبية الخاصة بالإسعافات الأولية."
                        else:
                            refusal_reason = "عذراً، المعلومات الطبية المتاحة غير كافية لتقديم إجابة موثوقة في الإسعافات الأولية."

                return ClinicalLLMResponse(
                    is_in_scope=is_in_scope,
                    is_knowledge_sufficient=is_knowledge_sufficient,
                    answer=answer,
                    citations=citations_list,
                    refusal_reason=refusal_reason,
                    provider="gemini",
                    model_name=self.model_name,
                    filtered_chunks_count=len(filtered_docs),
                )

        except Exception as e:
            logger.error(f"Failed to generate response via Gemini: {e}")
            return ClinicalLLMResponse(
                is_in_scope=True,
                is_knowledge_sufficient=False,
                answer=None,
                citations=[],
                refusal_reason=f"حدث خطأ أثناء التواصل مع Gemini: {str(e)}",
                provider="gemini",
                model_name=self.model_name,
                filtered_chunks_count=len(filtered_docs),
            )

    async def check_health(self) -> bool:
        """Check if Gemini API Key is configured and service endpoint is reachable."""
        return bool(self.api_key)
