from schemas.documents import ParsedDocument, ParsedSection
from services.cleaning_service import CleaningService


def test_cleaning_service_removes_page_furniture():
    cleaning_service = CleaningService()

    doc = ParsedDocument(
        document_id="test_hash_123",
        title="NICE Guideline Test",
        total_pages=5,
        sections=[
            ParsedSection(page_no=1, section_name="Title", text="Page 1 of 5"),
            ParsedSection(page_no=1, section_name="Content", text="1.1 Clinical Recommendation Text"),
            ParsedSection(page_no=1, section_name="Footer", text="© NICE 2024. All rights reserved."),
            ParsedSection(page_no=2, section_name="Header", text="Downloaded from NICE Website"),
            ParsedSection(page_no=2, section_name="Content", text="1.2 Second Clinical Recommendation"),
        ]
    )

    cleaned = cleaning_service.clean_document(doc)

    texts = [s.text for s in cleaned.sections]
    assert "1.1 Clinical Recommendation Text" in texts
    assert "1.2 Second Clinical Recommendation" in texts
    assert "Page 1 of 5" not in texts
    assert "© NICE 2024. All rights reserved." not in texts
