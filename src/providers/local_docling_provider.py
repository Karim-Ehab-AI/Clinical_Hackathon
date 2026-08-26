import os
import gc
import logging
from pathlib import Path
from typing import List, Optional
import torch

from interfaces.document_parser import DocumentParser
from schemas.documents import (
    ParsedDocument,
    ParsedSection,
    ParsedTable,
    ParsedFigure,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class LocalDoclingProvider(DocumentParser):
    """Local In-Process Docling Document Parser.
    
    Parses PDF documents directly on the local machine/instance using Docling.
    Features:
      - Lazy loading of Docling DocumentConverter and pipeline options.
      - Extracts sections, structured tables, and figures.
      - Cleans memory after document processing.
    """

    def __init__(self, do_ocr: bool = settings.DOCLING_DO_OCR):
        self.do_ocr = do_ocr
        self._converter = None

    @property
    def converter(self):
        """Lazy load Docling DocumentConverter."""
        if self._converter is None:
            logger.info(f"📄 Initializing local Docling DocumentConverter (do_ocr={self.do_ocr})...")
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                pipeline_options = PdfPipelineOptions(
                    do_ocr=self.do_ocr,
                    do_table_structure=True,
                )
                self._converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )
                logger.info("✅ Local Docling DocumentConverter initialized successfully!")
            except Exception as e:
                logger.error(f"❌ Failed to initialize local Docling converter: {e}", exc_info=True)
                raise RuntimeError(
                    f"Could not initialize local Docling converter. Ensure 'docling' is installed. Error: {e}"
                )
        return self._converter

    def convert_document(self, file_path: str):
        """Convert a PDF file path and return the Docling ConversionResult object."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            result = self.converter.convert(str(path_obj))
            return result
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def parse_pdf(self, file_path: str, document_id: str, document_title: str) -> ParsedDocument:
        """Parse PDF document directly in-process and return ParsedDocument schema."""
        logger.info(f"Converting PDF locally via Docling: {file_path}")
        conv_result = self.convert_document(file_path)
        doc = conv_result.document

        sections: List[ParsedSection] = []
        tables: List[ParsedTable] = []
        figures: List[ParsedFigure] = []

        total_pages = getattr(doc, "num_pages", 1)

        # Extract docling body texts / sections
        for idx, item in enumerate(doc.texts):
            page_no = 1
            if item.prov and len(item.prov) > 0 and item.prov[0].page_no is not None:
                page_no = item.prov[0].page_no

            sections.append(
                ParsedSection(
                    page_no=page_no,
                    section_name=getattr(item, "label", "text") or "text",
                    text=item.text,
                )
            )

        # Extract structured tables
        for tbl in doc.tables:
            page_no = 1
            if tbl.prov and len(tbl.prov) > 0 and tbl.prov[0].page_no is not None:
                page_no = tbl.prov[0].page_no

            headers = []
            rows = []
            try:
                df = tbl.export_to_dataframe()
                headers = list(df.columns)
                rows = df.values.tolist()
            except Exception:
                pass

            tables.append(
                ParsedTable(
                    page_no=page_no,
                    caption=getattr(tbl, "caption", "") or "",
                    headers=[str(h) for h in headers],
                    rows=[[str(c) for c in r] for r in rows],
                    text_content=tbl.export_to_markdown() if hasattr(tbl, "export_to_markdown") else "",
                )
            )

        # Extract figures / pictures
        for pic in getattr(doc, "pictures", []):
            page_no = 1
            if pic.prov and len(pic.prov) > 0 and pic.prov[0].page_no is not None:
                page_no = pic.prov[0].page_no

            figures.append(
                ParsedFigure(
                    page_no=page_no,
                    caption=getattr(pic, "caption", "") or "",
                    text_content=getattr(pic, "text", "") or "",
                )
            )

        logger.info(
            f"✅ Local Docling parsing completed: {total_pages} pages, "
            f"{len(sections)} text items, {len(tables)} tables, {len(figures)} figures."
        )

        return ParsedDocument(
            document_id=document_id,
            title=document_title,
            total_pages=total_pages,
            sections=sections,
            tables=tables,
            figures=figures,
        )

