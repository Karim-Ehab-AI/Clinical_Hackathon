import os
import logging
from typing import List
import httpx
from interfaces.document_parser import DocumentParser
from schemas.documents import (
    ParsedDocument,
    ParsedSection,
    ParsedTable,
    ParsedFigure,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class DoclingProvider(DocumentParser):
    """Remote-only Docling Document Parser Provider delegating all layout parsing to Colab GPU service."""

    def __init__(self, api_url: str = settings.EMBEDDING_API_URL):
        self.api_url = api_url.rstrip("/")

    def parse_pdf(self, file_path: str, document_id: str, document_title: str) -> ParsedDocument:
        """Parse PDF document exclusively via Remote Colab GPU endpoint (POST /parse_pdf)."""
        url = f"{self.api_url}/parse_pdf"
        logger.info(f"Delegating PDF parsing exclusively to remote Colab GPU service: {url}")

        headers = {
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "ClinicalRAG-Client/1.0",
        }

        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/pdf")}
                # 10 minute timeout for large clinical documents
                with httpx.Client(timeout=600.0) as client:
                    response = client.post(url, files=files, headers=headers)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Remote Colab GPU parsing failed with HTTP status {response.status_code}: {response.text}"
                )

            data = response.json()

            # Ensure expected payload structure
            if "sections" not in data:
                raise ValueError(f"Malformed response from remote Colab parsing endpoint. Got: {list(data.keys())}")

            sections = []
            for sec in data.get("sections", []):
                sections.append(
                    ParsedSection(
                        page_no=int(sec.get("page_no", 1) or 1),
                        section_name=str(sec.get("section_name", "") or ""),
                        text=str(sec.get("text", "") or ""),
                    )
                )

            tables = []
            for tbl in data.get("tables", []):
                raw_headers = [str(h) for h in tbl.get("headers", []) if h is not None]
                raw_rows = [[str(c) for c in row] for row in tbl.get("rows", []) if isinstance(row, list)]
                tables.append(
                    ParsedTable(
                        page_no=int(tbl.get("page_no", 1) or 1),
                        caption=str(tbl.get("caption", "") or ""),
                        headers=raw_headers,
                        rows=raw_rows,
                        text_content=str(tbl.get("text_content", "") or ""),
                    )
                )

            figures = []
            for fig in data.get("figures", []):
                figures.append(
                    ParsedFigure(
                        page_no=int(fig.get("page_no", 1) or 1),
                        caption=str(fig.get("caption", "") or ""),
                        text_content=str(fig.get("text_content", "") or ""),
                    )
                )

            total_pages = int(data.get("total_pages", 1) or 1)

            logger.info(
                f"Remote Colab GPU parsing completed successfully: "
                f"{total_pages} pages, {len(sections)} sections, {len(tables)} tables, {len(figures)} figures."
            )

            return ParsedDocument(
                document_id=document_id,
                title=document_title,
                total_pages=total_pages,
                sections=sections,
                tables=tables,
                figures=figures,
            )


        except Exception as e:
            logger.error(f"Remote Colab GPU PDF parsing error: {e}", exc_info=True)
            raise RuntimeError(f"Docling PDF parsing failed on remote Colab GPU: {str(e)}")


