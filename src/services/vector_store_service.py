from typing import List
from interfaces.vector_store import VectorStore
from schemas.documents import DocumentChunk, EmbeddingResult


class VectorStoreService:
    """Service wrapping VectorStore operations obeying Dependency Inversion Principle."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def document_exists(self, document_id: str) -> bool:
        """Check if document exists in vector store."""
        return self.vector_store.document_exists(document_id)

    def store_chunks(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> int:
        """Store chunk points and named vectors into vector database."""
        return self.vector_store.upsert_points(chunks, embeddings)
