import os
import gc
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import httpx
import torch

from transformers import AutoTokenizer
from schemas.documents import ChunkMetadata, DocumentChunk, EmbeddingResult
from interfaces.embedding_provider import EmbeddingProvider
from config.settings import settings

logger = logging.getLogger(__name__)

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
    Unified Local & Remote Pipeline for Medical / Clinical RAG.
    
    Modes:
      - Local (AWS / In-process):
          1. Direct Docling PDF layout conversion & HybridChunker structure chunking in-process.
          2. Enrich with business & clinical metadata (content_role, content_type, token_count, hashes).
          3. Generate Dense (1024-dim) & Sparse (lexical weights) embeddings via LocalEmbeddingProvider.
      - Remote (Colab Fallback):
          1. Upload PDF to remote /chunk_pdf endpoint.
          2. Enrich metadata locally.
          3. POST to remote /embed endpoint.
    """

    def __init__(
        self,
        tokenizer_name: str = settings.EMBEDDING_MODEL,
        source_type: str = "clinical_guideline",
        document_version: str = "2025",
        language: str = "en",
        chunk_max_tokens: int = 512,
    ):
        self.tokenizer_name = tokenizer_name
        self.source_type = source_type
        self.document_version = document_version
        self.language = language
        self.chunk_max_tokens = chunk_max_tokens
        self._tokenizer = None
        self._docling_converter = None
        self._docling_chunker = None

    @property
    def tokenizer(self):
        """Lazy load local AutoTokenizer for token counting."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except Exception as e:
                logger.warning(f"Could not load AutoTokenizer '{self.tokenizer_name}': {e}. Using basic word count.")
                self._tokenizer = None
        return self._tokenizer

    def _get_docling_components(self):
        """Lazy load Docling DocumentConverter and HybridChunker."""
        if self._docling_converter is None or self._docling_chunker is None:
            logger.info("Initializing in-process Docling DocumentConverter and HybridChunker...")
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
                from docling.chunking import HybridChunker

                pipeline_options = PdfPipelineOptions(
                    do_ocr=settings.DOCLING_DO_OCR,
                    do_table_structure=True,
                )
                self._docling_converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )

                hf_tok = HuggingFaceTokenizer.from_pretrained(self.tokenizer_name)
                self._docling_chunker = HybridChunker(tokenizer=hf_tok, max_tokens=self.chunk_max_tokens)
                logger.info("✅ In-process Docling components initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize Docling components: {e}", exc_info=True)
                raise RuntimeError(
                    f"Could not initialize local Docling parser or chunker. Ensure 'docling' and 'docling-core[chunking]' are installed. Error: {e}"
                )
        return self._docling_converter, self._docling_chunker

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
            logger.warning(f"⚠️ Chunk has mixed labels ({doc_item_labels}). Falling back to 'text'.")
            return "text"

        return "text"

    def _calculate_token_count(self, text: str) -> int:
        """Token count of text using BGE-M3 tokenizer locally."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        return len(text.split())

    async def process_pdf(
        self,
        pdf_path: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> List[DocumentChunk]:
        """Unified entrypoint: processes PDF locally or remotely based on configuration."""
        if settings.DOCLING_PROVIDER_TYPE == "remote" and settings.EMBEDDING_API_URL:
            logger.info(f"Delegating PDF processing to remote URL: {settings.EMBEDDING_API_URL}")
            return await self.process_pdf_remote(pdf_path=pdf_path, remote_base_url=settings.EMBEDDING_API_URL)

        logger.info(f"Processing PDF locally in-process: {pdf_path}")
        return await self.process_pdf_local(pdf_path=pdf_path, embedding_provider=embedding_provider)

    async def process_pdf_local(
        self,
        pdf_path: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> List[DocumentChunk]:
        """
        Execute full local in-process PDF chunking and embedding pipeline (AWS-ready).
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        # 1. Compute local SHA-256 document_id
        pdf_bytes = path_obj.read_bytes()
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        document_title = path_obj.stem
        source_name = path_obj.name

        logger.info(f"📄 Local PDF Processing started for '{source_name}' (ID: {document_id[:10]}...)")

        # 2. Local Docling Conversion and Chunking
        docling_converter, chunker = self._get_docling_components()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            conv_result = docling_converter.convert(str(path_obj))
            doc = conv_result.document
            chunks_gen = list(chunker.chunk(dl_doc=doc))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        total_pages = getattr(doc, "num_pages", 1)
        logger.info(f"✅ Docling generated {len(chunks_gen)} raw structural chunks across {total_pages} pages.")

        if not chunks_gen:
            logger.warning(f"No structural chunks created by Docling for '{source_name}'.")
            return []

        # 3. Build Metadata per Chunk
        chunk_objects_metadata: List[Dict[str, Any]] = []
        texts_for_embedding: List[str] = []

        for i, chunk in enumerate(chunks_gen):
            text = chunk.text or ""
            if not text.strip():
                continue

            contextualized_text = chunker.contextualize(chunk) or text
            chunk_id = f"{document_id}_chunk_{i:05d}"

            pages = sorted(list(set(
                prov.page_no for item in chunk.meta.doc_items for prov in item.prov if prov.page_no is not None
            )))
            labels = sorted(list(set(
                str(item.label) for item in chunk.meta.doc_items
            )))
            headings = chunk.meta.headings if chunk.meta.headings else []

            content_type = self._map_content_type(labels)
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
                chunk_index=i,
                content_hash=content_hash,
            )

            chunk_objects_metadata.append({
                "chunk_id": chunk_id,
                "text": text,
                "contextualized_text": contextualized_text,
                "metadata": meta,
            })
            texts_for_embedding.append(contextualized_text)

        # 4. Generate Embeddings locally via EmbeddingProvider
        if embedding_provider is None:
            from providers.local_embedding_provider import LocalEmbeddingProvider
            embedding_provider = LocalEmbeddingProvider()

        logger.info(f"🧠 Generating dense+sparse embeddings for {len(texts_for_embedding)} chunks...")
        embedding_results: List[EmbeddingResult] = await embedding_provider.embed_documents(texts_for_embedding)

        # 5. Assemble final DocumentChunk objects
        final_document_chunks: List[DocumentChunk] = []
        for idx, item in enumerate(chunk_objects_metadata):
            dense_vec = None
            sparse_indices = None
            sparse_values = None

            if idx < len(embedding_results):
                emb_res = embedding_results[idx]
                dense_vec = emb_res.dense
                sparse_indices = emb_res.sparse_indices
                sparse_values = emb_res.sparse_values

            final_chunk = DocumentChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                metadata=item["metadata"],
                dense_vector=dense_vec,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
            )
            final_document_chunks.append(final_chunk)

        logger.info(f"🚀 Successfully generated {len(final_document_chunks)} ready-to-index DocumentChunk objects locally!")
        return final_document_chunks

    async def process_pdf_remote(
        self,
        pdf_path: str,
        remote_base_url: str,
    ) -> List[DocumentChunk]:
        """
        Execute full remote-chunking and embedding pipeline for a PDF document (Colab fallback).
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        pdf_bytes = path_obj.read_bytes()
        document_id = hashlib.sha256(pdf_bytes).hexdigest()
        document_title = path_obj.stem
        source_name = path_obj.name

        remote_base_url = remote_base_url.rstrip("/")
        chunk_pdf_url = f"{remote_base_url}/chunk_pdf"
        embed_url = f"{remote_base_url}/embed"

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

        logger.info(f"🌐 Sending {len(texts_for_embedding)} contextualized texts to remote embed endpoint '{embed_url}'...")
        async with httpx.AsyncClient(timeout=180.0) as client:
            res = await client.post(embed_url, json={"texts": texts_for_embedding})
            if res.status_code != 200:
                raise RuntimeError(f"Remote /embed endpoint failed ({res.status_code}): {res.text}")
            
            embed_response = res.json()

        dense_vectors = embed_response.get("dense", []) if isinstance(embed_response, dict) else []
        sparse_objects = embed_response.get("sparse", []) if isinstance(embed_response, dict) else []
        legacy_embeddings = embed_response if isinstance(embed_response, list) else embed_response.get("embeddings", [])

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
