from typing import List, Dict
from collections import Counter
from schemas.documents import ParsedDocument, ParsedSection
from utils.clinical_regex import is_page_number_text, is_boilerplate_text


class CleaningService:
    """Cleans document furniture (headers, footers, page numbers, repeated download notices)
    without deleting legitimate section headings or repeating clinical terms.
    """

    def clean_document(self, doc: ParsedDocument) -> ParsedDocument:
        """Clean page furniture from parsed document sections."""
        if not doc.sections:
            return doc

        # 1. Identify cross-page repeated lines (candidate headers/footers)
        line_occurrences: Counter = Counter()
        for sec in doc.sections:
            clean_text = sec.text.strip()
            if clean_text and len(clean_text) < 150:  # Short text candidates
                line_occurrences[clean_text] += 1

        # A line repeating on > 3 pages or > 20% of pages is a strong header/footer candidate
        threshold = max(3, int(doc.total_pages * 0.2)) if doc.total_pages > 0 else 3
        repeating_furniture = {
            text for text, count in line_occurrences.items()
            if count >= threshold and (is_boilerplate_text(text) or is_page_number_text(text))
        }

        # 2. Filter sections
        cleaned_sections: List[ParsedSection] = []
        for sec in doc.sections:
            text = sec.text.strip()

            # Skip empty sections
            if not text:
                continue

            # Skip standalone page numbers
            if is_page_number_text(text):
                continue

            # Skip recognized repeating furniture
            if text in repeating_furniture:
                continue

            # Skip explicit copyright/boilerplate matches
            if is_boilerplate_text(text) and len(text) < 120:
                continue

            cleaned_sections.append(sec)

        doc.sections = cleaned_sections
        return doc
