from typing import Optional, List
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Chunk metadata schema for First Aid / Clinical RAG System."""

    # Provenance
    chunk_id: str
    document_id: str
    document_title: str = ""
    source: str = ""
    source_type: str = "clinical_guideline"
    document_version: str = "2025"
    language: str = "en"

    # Structure & location
    document_part: Optional[str] = None   # first_aid | resuscitation | education
    content_type: str = "text"            # text | table | figure
    pdf_pages: List[int] = Field(default_factory=list)
    heading_path: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    subsection: Optional[str] = None
    content_role: Optional[str] = None    # key_action | first_aid_steps | caution | ...

    # Chunk-level
    token_count: int = 0
    chunk_index: int = 0
    content_hash: str = ""

    # Legacy fallback fields for backward compatibility
    pdf_page: int = 1
    document_page: int = 1
    recommendation_id: Optional[str] = None
    recommendation_class: Optional[str] = None
    evidence_level: Optional[str] = None
    is_table: bool = False
    is_figure: bool = False


class DocumentChunk(BaseModel):
    """Final chunk object ready for vector database indexing (Qdrant / pgvector)."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_vector: Optional[List[float]] = None
    sparse_indices: Optional[List[int]] = None
    sparse_values: Optional[List[float]] = None


class EmbeddingResult(BaseModel):
    """Atomic dense and sparse embedding payload produced per text item."""

    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]


class RemoteChunkResponse(BaseModel):
    """Raw chunk payload returned from remote /chunk_pdf endpoint."""

    chunk_index: int
    text: str
    contextualized_text: str
    headings: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    doc_item_labels: List[str] = Field(default_factory=list)


class RemotePDFChunkingResult(BaseModel):
    """Response returned from remote /chunk_pdf endpoint."""

    total_pages: int
    chunk_count: int
    chunks: List[RemoteChunkResponse]


class ParsedTable(BaseModel):
    """Parsed table representation for document parser compatibility."""

    page_no: int
    caption: str = ""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    text_content: str = ""


class ParsedFigure(BaseModel):
    """Parsed figure representation for document parser compatibility."""

    page_no: int
    caption: str = ""
    text_content: str = ""


class ParsedSection(BaseModel):
    """Parsed section/text node representation for document parser compatibility."""

    page_no: int
    section_name: str = ""
    subsection_name: str = ""
    text: str = ""
    bounding_box: Optional[dict] = None


class ParsedDocument(BaseModel):
    """Internal normalized output from the Document Parser."""

    document_id: str
    title: str
    total_pages: int
    sections: List[ParsedSection] = Field(default_factory=list)
    tables: List[ParsedTable] = Field(default_factory=list)
    figures: List[ParsedFigure] = Field(default_factory=list)
