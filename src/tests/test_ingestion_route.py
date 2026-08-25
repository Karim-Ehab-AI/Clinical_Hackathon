import io
import pytest
import httpx
from fastapi.testclient import TestClient
from main import app
from routes.ingestion import get_ingestion_controller
from controllers.ingestion_controller import IngestionController
from services.document_service import DocumentService
from schemas.documents import ParsedDocument, ParsedSection, EmbeddingResult


class DummyParser:
    def parse_pdf(self, file_path, document_id, document_title):
        return ParsedDocument(
            document_id=document_id,
            title=document_title,
            total_pages=1,
            sections=[
                ParsedSection(
                    page_no=1,
                    section_name="Test Section",
                    text="Recommendation 1.5.7 Offer metformin as first-line treatment (Class I, Level A).",
                )
            ],
        )


class DummyEmbeddingProvider:
    async def embed_documents(self, texts):
        return [
            EmbeddingResult(
                dense=[0.05] * 1024,
                sparse_indices=[1, 2],
                sparse_values=[0.1, 0.2],
            )
            for _ in texts
        ]

    async def embed_query(self, text):
        return EmbeddingResult(dense=[0.05] * 1024, sparse_indices=[1], sparse_values=[0.1])

    async def check_health(self):
        return True


class DummyVectorStore:
    def __init__(self):
        self.stored = False

    def ensure_collection(self):
        pass

    def document_exists(self, document_id):
        return False

    def upsert_points(self, chunks, embeddings):
        self.stored = True
        return len(chunks)


def override_ingestion_controller():
    dummy_parser = DummyParser()
    dummy_emb = DummyEmbeddingProvider()
    dummy_vs = DummyVectorStore()

    from services.vector_store_service import VectorStoreService
    vs_service = VectorStoreService(dummy_vs)

    doc_service = DocumentService(
        parser=dummy_parser,
        embedding_provider=dummy_emb,
        vector_store_service=vs_service,
    )
    return IngestionController(document_service=doc_service)


app.dependency_overrides[get_ingestion_controller] = override_ingestion_controller

client = TestClient(app)


def test_upload_pdf_route():
    pdf_content = b"%PDF-1.4 Mock PDF Content For Testing Upload Route"
    files = {"file": ("test_guideline.pdf", io.BytesIO(pdf_content), "application/pdf")}

    response = client.post("/api/v1/ingestion/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["document_id"]) == 64  # SHA-256 hash
    assert data["filename"] == "test_guideline.pdf"
    assert data["chunks_created"] == 1
    assert data["vectors_stored"] == 1


def test_upload_non_pdf_fails():
    files = {"file": ("test.txt", io.BytesIO(b"Text content"), "text/plain")}
    response = client.post("/api/v1/ingestion/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]
