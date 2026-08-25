from abc import ABC, abstractmethod
from schemas.documents import ParsedDocument


class DocumentParser(ABC):
    """Abstract Interface for Document Parsing providers (e.g. Docling)."""

    @abstractmethod
    def parse_pdf(self, file_path: str, document_id: str, document_title: str) -> ParsedDocument:
        """Parse a PDF document into a structured ParsedDocument representation."""
        pass
