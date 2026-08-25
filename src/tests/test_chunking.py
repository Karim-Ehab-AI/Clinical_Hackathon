from schemas.documents import ParsedDocument, ParsedSection, ParsedTable
from services.chunking_service import ChunkingService
from utils.clinical_regex import (
    extract_nice_recommendation_id,
    extract_esc_metadata,
)


def test_nice_recommendation_regex():
    text1 = "Recommendation 1.5.7 Offer metformin as first-line treatment."
    text2 = "Recommendation 1.5.7 to 1.5.9 for adults with type 2 diabetes."
    text3 = "See section 1.2.3 for detail."

    assert extract_nice_recommendation_id(text1) == "1.5.7"
    assert extract_nice_recommendation_id(text2) == "1.5.7 to 1.5.9"
    assert extract_nice_recommendation_id(text3) == "1.2.3"


def test_esc_class_level_regex():
    text1 = "SGLT2 inhibitors are recommended in patients with HF (Class I, Level A)."
    text2 = "Consider beta-blocker therapy (Class IIa, Level B)."
    text3 = "Routine use is not recommended (Class III, Level C)."

    c1, l1 = extract_esc_metadata(text1)
    assert c1 == "I" and l1 == "A"

    c2, l2 = extract_esc_metadata(text2)
    assert c2 == "IIa" and l2 == "B"

    c3, l3 = extract_esc_metadata(text3)
    assert c3 == "III" and l3 == "C"


def test_chunking_service_creates_chunks_with_metadata():
    service = ChunkingService()

    doc = ParsedDocument(
        document_id="doc_hash_999",
        title="ESC Heart Failure Guidelines",
        total_pages=2,
        sections=[
            ParsedSection(
                page_no=1,
                section_name="Pharmacotherapy",
                text="Recommendation 1.5.7 to 1.5.9: SGLT2 inhibitors are indicated for HF (Class I, Level A).",
            )
        ],
        tables=[
            ParsedTable(
                page_no=2,
                caption="ESC Class of Recommendation Summary",
                headers=["Intervention", "Class", "Level"],
                rows=[
                    ["Dapagliflozin 10mg", "Class I", "Level A"],
                    ["Empagliflozin 10mg", "Class I", "Level A"],
                ],
                text_content="Dapagliflozin 10mg Class I Level A",
            )
        ]
    )

    chunks = service.create_chunks(doc)

    assert len(chunks) == 2

    # Section Chunk
    sec_chunk = chunks[0]
    assert sec_chunk.metadata.recommendation_id == "1.5.7 to 1.5.9"
    assert sec_chunk.metadata.recommendation_class == "I"
    assert sec_chunk.metadata.evidence_level == "A"
    assert sec_chunk.metadata.is_table is False

    # Table Chunk
    tbl_chunk = chunks[1]
    assert tbl_chunk.metadata.is_table is True
    assert tbl_chunk.metadata.recommendation_class == "I"
    assert tbl_chunk.metadata.evidence_level == "A"
    assert "Header: Intervention | Class | Level" in tbl_chunk.text
