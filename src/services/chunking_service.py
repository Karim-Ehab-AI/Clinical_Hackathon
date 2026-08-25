import uuid
from typing import List, Optional
from transformers import AutoTokenizer

from schemas.documents import ParsedDocument, DocumentChunk, ChunkMetadata
from utils.clinical_regex import (
    extract_nice_recommendation_id,
    extract_esc_metadata,
)
from config.settings import settings


class ChunkingService:
    """Structure-aware, semantic-aware, token-aware chunking service.
    Configured with BAAI/bge-m3 tokenizer.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load HuggingFace BAAI/bge-m3 tokenizer."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception:
                # Fallback basic tokenizer if offline/unavailable during tests
                self._tokenizer = None
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens using BAAI/bge-m3 tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback estimation if offline
        return len(text.split())

    def create_chunks(self, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        """Convert ParsedDocument into enriched DocumentChunk objects."""
        chunks: List[DocumentChunk] = []

        # 1. Chunk Text Sections
        for sec in parsed_doc.sections:
            sec_text = sec.text.strip()
            if not sec_text:
                continue

            # Check for NICE recommendation ID and ESC Class/Level metadata
            rec_id = extract_nice_recommendation_id(sec_text)
            esc_class, esc_level = extract_esc_metadata(sec_text)

            token_count = self.count_tokens(sec_text)

            # If within token target (~300-600 tokens), keep as single chunk
            if token_count <= 600:
                chunk_id = str(uuid.uuid4())
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=parsed_doc.document_id,
                    document_title=parsed_doc.title,
                    source=f"{settings.ASSETS_DIR}/{parsed_doc.document_id}.pdf",
                    pdf_page=sec.page_no,
                    document_page=sec.page_no,
                    section=sec.section_name,
                    subsection=sec.subsection_name,
                    recommendation_id=rec_id,
                    recommendation_class=esc_class,
                    evidence_level=esc_level,
                    content_type="text",
                    is_table=False,
                    is_figure=False,
                )
                chunks.append(DocumentChunk(chunk_id=chunk_id, text=sec_text, metadata=meta))
            else:
                # Split longer section into token-bounded sub-chunks while maintaining section context
                sub_texts = self._split_text_by_tokens(sec_text, target_tokens=450)
                for sub_t in sub_texts:
                    sub_rec_id = extract_nice_recommendation_id(sub_t) or rec_id
                    sub_class, sub_level = extract_esc_metadata(sub_t)
                    sub_class = sub_class or esc_class
                    sub_level = sub_level or esc_level

                    chunk_id = str(uuid.uuid4())
                    meta = ChunkMetadata(
                        chunk_id=chunk_id,
                        document_id=parsed_doc.document_id,
                        document_title=parsed_doc.title,
                        source=f"{settings.ASSETS_DIR}/{parsed_doc.document_id}.pdf",
                        pdf_page=sec.page_no,
                        document_page=sec.page_no,
                        section=sec.section_name,
                        subsection=sec.subsection_name,
                        recommendation_id=sub_rec_id,
                        recommendation_class=sub_class,
                        evidence_level=sub_level,
                        content_type="text",
                        is_table=False,
                        is_figure=False,
                    )
                    chunks.append(DocumentChunk(chunk_id=chunk_id, text=sub_t, metadata=meta))

        # 2. Chunk Tables (preserving table header rows in each chunk)
        for tbl in parsed_doc.tables:
            header_str = " | ".join(tbl.headers) if tbl.headers else ""
            table_chunks = self._chunk_table(tbl, header_str, parsed_doc)
            chunks.extend(table_chunks)

        # 3. Chunk Figures
        for fig in parsed_doc.figures:
            if not fig.text_content:
                continue
            chunk_id = str(uuid.uuid4())
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=parsed_doc.document_id,
                document_title=parsed_doc.title,
                source=f"{settings.ASSETS_DIR}/{parsed_doc.document_id}.pdf",
                pdf_page=fig.page_no,
                document_page=fig.page_no,
                section="Figures",
                content_type="figure",
                is_table=False,
                is_figure=True,
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=fig.text_content, metadata=meta))

        return chunks

    def _split_text_by_tokens(self, text: str, target_tokens: int = 450) -> List[str]:
        """Split text into sentences/paragraphs bounded by target token limit."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_tokens = 0

        for p in paragraphs:
            p_tokens = self.count_tokens(p)
            if current_tokens + p_tokens > target_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_tokens = p_tokens
            else:
                current_chunk.append(p)
                current_tokens += p_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _chunk_table(self, tbl, header_str: str, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        """Chunk table preserving column headers and extracting ESC Class/Level metadata from table structure."""
        chunks = []
        rows_per_chunk = 15  # Keep table chunk compact

        if not tbl.rows:
            if tbl.text_content:
                chunk_id = str(uuid.uuid4())
                esc_class, esc_level = extract_esc_metadata(tbl.text_content)
                meta = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=parsed_doc.document_id,
                    document_title=parsed_doc.title,
                    source=f"{settings.ASSETS_DIR}/{parsed_doc.document_id}.pdf",
                    pdf_page=tbl.page_no,
                    document_page=tbl.page_no,
                    section="Tables",
                    recommendation_class=esc_class,
                    evidence_level=esc_level,
                    content_type="table",
                    is_table=True,
                    is_figure=False,
                )
                chunks.append(DocumentChunk(chunk_id=chunk_id, text=tbl.text_content, metadata=meta))
            return chunks

        for i in range(0, len(tbl.rows), rows_per_chunk):
            row_batch = tbl.rows[i : i + rows_per_chunk]
            formatted_rows = [" | ".join(r) for r in row_batch]

            if header_str:
                chunk_text = f"Table Caption: {tbl.caption}\nHeader: {header_str}\n" + "\n".join(formatted_rows)
            else:
                chunk_text = f"Table Caption: {tbl.caption}\n" + "\n".join(formatted_rows)

            esc_class, esc_level = extract_esc_metadata(chunk_text)
            rec_id = extract_nice_recommendation_id(chunk_text)

            chunk_id = str(uuid.uuid4())
            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=parsed_doc.document_id,
                document_title=parsed_doc.title,
                source=f"{settings.ASSETS_DIR}/{parsed_doc.document_id}.pdf",
                pdf_page=tbl.page_no,
                document_page=tbl.page_no,
                section="Tables",
                recommendation_id=rec_id,
                recommendation_class=esc_class,
                evidence_level=esc_level,
                content_type="table",
                is_table=True,
                is_figure=False,
            )
            chunks.append(DocumentChunk(chunk_id=chunk_id, text=chunk_text, metadata=meta))

        return chunks
