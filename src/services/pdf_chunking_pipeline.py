import os
import hashlib
import logging
import httpx
from pathlib import Path
from typing import List, Optional, Dict, Any

from transformers import AutoTokenizer
from schemas.documents import ChunkMetadata, DocumentChunk, RemotePDFChunkingResult

logger = logging.getLogger(__name__)

import re

# Known medical section keywords for rule-based content_role mapping
CONTENT_ROLE_MAP = {
    "key_action": ["key action", "key actions", "action required", "immediate action", "action", "actions"],
    "access_help": ["access help", "call emergency", "emergency help", "when to call", "emergency", "assistance", "help"],
    "caution": ["caution", "warning", "warnings", "risk", "risks", "danger", "dangers", "contraindication", "contraindications", "do not"],
    "recovery": ["recovery", "aftercare", "post-care", "follow-up", "monitoring"],
    "first_aid_steps": ["first aid", "first-aid", "first aid steps", "initial steps", "step", "steps"],
    "good_practice": ["good practice", "best practice", "practice point", "practice points", "good_practice"],
    "education": ["education", "prevention", "training", "awareness"],
    "scientific_foundation": ["scientific foundation", "scientific", "evidence", "rationale", "rationales", "foundation", "foundations"],
    "introduction": ["introduction", "overview", "background", "scope", "about this guideline", "about"],
}


class PDFChunkingPipeline:
    """
    Local-Remote Hybrid Pipeline for Medical / First Aid RAG.
    
    Architecture:
      1. Local: Compute PDF SHA-256 document_id & upload PDF to POST {remote_base_url}/chunk_pdf
      2. Remote: Docling parses PDF & returns structural chunks (text, contextualized_text, headings, pages, doc_item_labels)
      3. Local: Build complete business metadata per chunk (content_role, content_type, token_count, content_hash, etc.)
      4. Remote: POST contextualized_text strings to {remote_base_url}/embed to receive dense & sparse vectors
      5. Combine into final DocumentChunk objects ready for indexing into Qdrant/pgvector.
    """

    def __init__(
        self,
        tokenizer_name: str = "BAAI/bge-m3",
        source_type: str = "clinical_guideline",
        document_version: str = "2025",
        language: str = "en",
    ):
        self.tokenizer_name = tokenizer_name
        self.source_type = source_type
        self.document_version = document_version
        self.language = language

        logger.info(f"Initializing local HuggingFace tokenizer '{tokenizer_name}' for token counting...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _derive_content_role(self, heading_path: List[str]) -> Optional[str]:
        """Rule-based content_role lookup against the last 1-2 entries of heading_path."""
        if not heading_path:
            return None

        relevant_headers = heading_path[-2:]
        combined_header_text = " ".join(relevant_headers).lower()

        for role_name, keywords in CONTENT_ROLE_MAP.items():
            for kw in keywords:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, combined_header_text):
                    return role_name

        return None

    def _map_content_type(self, doc_item_labels: List[str]) -> str:
        """Map doc_item_labels to 'text' | 'table' | 'figure'. Fall back to 'text' if unclear."""
        if not doc_item_labels:
            return "text"

        labels_lower = [str(lbl).lower() for lbl in doc_item_labels]
        has_table = any("table" in lbl for lbl in labels_lower)
        has_figure = any("picture" in lbl or "figure" in lbl or "caption" in lbl for lbl in labels_lower)

        if has_table and not has_figure:
            return "table"
        if has_figure and not has_table:
            return "figure"
        if has_table and has_figure:
            logger.warning(f"⚠️ Chunk has mixed labels ({doc_item_labels}). Falling back to 'text' for manual review.")
            return "text"

        return "text"

    def _calculate_token_count(self, text: str) -> int:
        """Token count of text using BGE-M3 tokenizer locally."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    async def process_pdf_remote(
        self,
        pdf_path: str,
        remote_base_url: str,
    ) -> List[DocumentChunk]:
        """
        Execute full remote-chunking and embedding pipeline for a PDF document.
        
        Params:
          pdf_path: Local path to the PDF file
          remote_base_url: Remote microservice base URL (e.g., http://remote-service:8000 or ngrok URL)
        Returns:
          List of DocumentChunk objects ready for vector database indexing.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        # 1. Compute local SHA-256 document_id
        pdf_bytes = path_obj.read_bytes()
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        document_title = path_obj.stem
        source_name = path_obj.name

        remote_base_url = remote_base_url.rstrip("/")
        chunk_pdf_url = f"{remote_base_url}/chunk_pdf"
        embed_url = f"{remote_base_url}/embed"

        # 2. Upload PDF to remote /chunk_pdf endpoint
        logger.info(f"📄 Uploading '{source_name}' (ID: {document_id[:10]}...) to remote chunker '{chunk_pdf_url}'...")
        async with httpx.AsyncClient(timeout=180.0) as client:
            files = {"file": (source_name, pdf_bytes, "application/pdf")}
            res = await client.post(chunk_pdf_url, files=files)
            if res.status_code != 200:
                raise RuntimeError(f"Remote /chunk_pdf failed ({res.status_code}): {res.text}")
            
            chunk_data = res.json()

        raw_chunks = chunk_data.get("chunks", [])
        total_pages = chunk_data.get("total_pages", 1)
        chunk_count = chunk_data.get("chunk_count", len(raw_chunks))
        logger.info(f"✅ Received {chunk_count} structural chunks across {total_pages} pages from remote service.")

        if not raw_chunks:
            logger.warning("No structural chunks returned from remote /chunk_pdf service.")
            return []

        # 3. Build local metadata per chunk
        chunk_objects_metadata: List[Dict[str, Any]] = []
        texts_for_embedding: List[str] = []

        for item in raw_chunks:
            chunk_idx = item.get("chunk_index", len(chunk_objects_metadata))
            text = item.get("text", "")
            contextualized_text = item.get("contextualized_text") or text

            if not text.strip():
                continue

            chunk_id = f"{document_id}_chunk_{chunk_idx:05d}"
            headings = item.get("headings", [])
            pages = item.get("pages", [])
            doc_item_labels = item.get("doc_item_labels", [])

            content_type = self._map_content_type(doc_item_labels)
            section = headings[0] if headings else None
            subsection = headings[1] if len(headings) > 1 else None
            content_role = self._derive_content_role(headings)

            token_count = self._calculate_token_count(text)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            first_page = pages[0] if (pages and len(pages) > 0) else 1
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                document_title=document_title,
                source=source_name,
                source_type=self.source_type,
                document_version=self.document_version,
                language=self.language,
                document_part=getattr(self, "document_part", None),
                content_type=content_type,
                pdf_pages=pages,
                pdf_page=first_page,
                document_page=first_page,
                heading_path=headings,
                section=section,
                subsection=subsection,
                content_role=content_role,
                token_count=token_count,
                chunk_index=chunk_idx,
                content_hash=content_hash,
            )

            chunk_objects_metadata.append({
                "chunk_id": chunk_id,
                "text": text,
                "contextualized_text": contextualized_text,
                "metadata": meta,
            })
            texts_for_embedding.append(contextualized_text)

        # 4. Batch contextualized texts and POST to /embed endpoint for dense + sparse vectors
        logger.info(f"🌐 Sending {len(texts_for_embedding)} contextualized texts to remote embed endpoint '{embed_url}'...")
        async with httpx.AsyncClient(timeout=180.0) as client:
            res = await client.post(embed_url, json={"texts": texts_for_embedding})
            if res.status_code != 200:
                raise RuntimeError(f"Remote /embed endpoint failed ({res.status_code}): {res.text}")
            
            embed_response = res.json()

        # Parse vectors array according to API Contract shape:
        # { "dense": [[...]], "sparse": [{"indices": [...], "values": [...]}], "dense_size": 1024 }
        dense_vectors = embed_response.get("dense", []) if isinstance(embed_response, dict) else []
        sparse_objects = embed_response.get("sparse", []) if isinstance(embed_response, dict) else []
        legacy_embeddings = embed_response if isinstance(embed_response, list) else embed_response.get("embeddings", [])

        # 5. Combine metadata + text + vectors into final DocumentChunk objects
        final_document_chunks: List[DocumentChunk] = []
        for idx, item in enumerate(chunk_objects_metadata):
            dense_vec = None
            sparse_indices = None
            sparse_values = None

            if dense_vectors and idx < len(dense_vectors):
                dense_vec = dense_vectors[idx]
                if sparse_objects and idx < len(sparse_objects):
                    sparse_info = sparse_objects[idx] or {}
                    sparse_indices = sparse_info.get("indices") or sparse_info.get("sparse_indices")
                    sparse_values = sparse_info.get("values") or sparse_info.get("sparse_values")
            elif legacy_embeddings and idx < len(legacy_embeddings):
                vec_data = legacy_embeddings[idx] or {}
                dense_vec = vec_data.get("dense")
                sparse_indices = vec_data.get("sparse_indices") or vec_data.get("indices")
                sparse_values = vec_data.get("sparse_values") or vec_data.get("values")

            final_chunk = DocumentChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item["metadata"],
                dense_vector=dense_vec,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
            )
            final_document_chunks.append(final_chunk)

        logger.info(f"🚀 Successfully generated {len(final_document_chunks)} ready-to-index DocumentChunk objects!")
        return final_document_chunks
