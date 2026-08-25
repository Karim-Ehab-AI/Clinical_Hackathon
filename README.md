# 🏥 Clinical RAG — Emergency First Aid Clinical Decision Support System

A modular, SOLID-compliant Clinical RAG (Retrieval-Augmented Generation) system for emergency First Aid guidance. Built with **FastAPI**, **Qdrant Vector Database**, **Docling**, **BAAI/bge-m3** (Dense + Sparse Embeddings), **bge-reranker-v2-m3** (Cross-Encoder Reranking), **Google Gemini LLM**, and a modern **React 19 / TanStack Start Frontend UI**.

---

## 📄 OpenAPI Specification (`openapi.json`)

> [!IMPORTANT]
> A complete, fully typed **OpenAPI 3.1.0** schema is provided in the repository at **[`openapi.json`](./openapi.json)**.
> 
> You can import `openapi.json` into Postman, Insomnia, or any API client.  
> Interactive documentation is also available out-of-the-box when running the backend:
> - 📍 **Swagger UI Docs:** `http://localhost:3000/docs`
> - 📍 **ReDoc Interactive Docs:** `http://localhost:3000/redoc`

---

## 🚀 Quick Start & Setup Guide / طريقة التشغيل

### 📋 Prerequisites
- **Docker Desktop** (running locally)
- **Python 3.10+** (Conda environment `clinical-rag` recommended)
- **Node.js 18+** & `npm`

---

### 1️⃣ Step 1: Launch Vector Database (Qdrant)
Start the Qdrant vector database container via Docker Compose from the project root:

```bash
docker compose up -d
```
> 📍 **Qdrant Dashboard & API:** `http://localhost:6333`

---

### 2️⃣ Step 2: Configure Environment Variables
Copy `.env.example` to `.env` in the root directory:

```bash
cp .env.example .env
```

Configure your `.env` settings:
```ini
APP_NAME=clinical-rag
DEBUG=true

# Qdrant Settings
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=clinical_documents

# Remote GPU Embedding & Reranker Service (Google Colab / ngrok)
EMBEDDING_API_URL=https://your-ngrok-domain.ngrok-free.app
RERANKER_ENABLED=true
RERANKER_API_URL=https://your-ngrok-domain.ngrok-free.app

# Gemini LLM Provider Key
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
MIN_SIMILARITY_SCORE_THRESHOLD=80.0
```

---

### 3️⃣ Step 3: Start FastAPI Backend Server
Navigate to `src/` and launch the Uvicorn server on port **3000**:

```bash
# Activate python environment
conda activate clinical-rag

# Run backend server
cd src
uvicorn main:app --host 0.0.0.0 --port 3000 --reload
```
> 📍 **Backend Base URL:** `http://localhost:3000`  
> 📍 **Swagger API Documentation:** `http://localhost:3000/docs`

---

### 4️⃣ Step 4: Start React Frontend UI
In a **new terminal**, navigate to `src/UI` and run the development server:

```bash
cd src/UI

# Install dependencies (first time only)
npm install

# Start Vite dev server
npm run dev
```
> 📍 **Frontend Web App:** `http://localhost:8080`

---

## 📡 API Endpoints Specification

The table below summarizes all active API routes in the system:

| Method | Endpoint | Description | Request Payload | Response Model |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/generation/generate` | Emergency First Aid LLM decision support with citations | `GenerateRequest` (JSON) | `GenerateResponse` (JSON) |
| `POST` | `/api/v1/ingestion/upload` | Upload & ingest clinical PDF document into Qdrant | `multipart/form-data` (File) | `IngestionResponse` (JSON) |
| `POST` | `/api/v1/retrieval/search` | Hybrid search (Dense+Sparse BGE-M3) + RRF + Reranker | `SearchRequest` (JSON) | `SearchResponse` (JSON) |
| `POST` | `/api/v1/retrieval/rerank` | Standalone cross-encoder reranking endpoint | `RerankRequest` (JSON) | `RerankResponse` (JSON) |
| `GET` | `/health` | API Health & status check | None | Status JSON |

---

### Detailed Endpoint Breakdown

#### 1. 🤖 Generation Endpoint (`POST /api/v1/generation/generate`)
Executes the end-to-end First Aid Clinical Decision Support pipeline.
- Performs hybrid search against vector database.
- Filters candidate chunks by a strict **80% similarity threshold**.
- Enforces double guardrails: **Is In Scope** (First Aid) & **Is Knowledge Sufficient**.
- Generates concise bullet-point instructions in the user's query language with exact chunk citations.

**Request Body (`application/json`):**
```json
{
  "query": "What should I do immediately for a second-degree burn on the arm?"
}
```

**Response (`200 OK`):**
```json
{
  "query": "What should I do immediately for a second-degree burn on the arm?",
  "retrieved_chunks_count": 10,
  "filtered_chunks_count": 3,
  "result": {
    "is_in_scope": true,
    "is_knowledge_sufficient": true,
    "answer": "• Cool the burn under cool running water for at least 20 minutes [1].\n• Remove any clothing or jewelry near the burned area unless stuck to skin [1].\n• Cover the burn loosely with sterile non-adherent dressing or clean plastic wrap [1].",
    "citations": [
      {
        "chunk_id": "doc_123_chunk_4",
        "source": "First_Aid_Manual.pdf",
        "pdf_page": 45,
        "section": "Thermal Burns",
        "recommendation_id": "REC-BURN-01",
        "source_text": "Cool the burn immediately under cool running water...",
        "score": 0.895,
        "percentage_score": 89.5
      }
    ],
    "refusal_reason": null,
    "provider": "gemini",
    "model_name": "gemini-1.5-flash",
    "filtered_chunks_count": 3
  }
}
```

---

#### 2. 📄 Ingestion Upload Endpoint (`POST /api/v1/ingestion/upload`)
Uploads a clinical PDF document, parses it with **Docling**, cleans headers/footers, tokenizes chunks, extracts NICE/ESC metadata, computes BAAI/bge-m3 dense & sparse vectors, and indexes them into Qdrant.

**Request Body (`multipart/form-data`):**
- `file`: PDF binary file (`.pdf`)

**Response (`200 OK`):**
```json
{
  "status": "success",
  "document_id": "a1b2c3d4e5f6...",
  "filename": "First_Aid_Manual.pdf",
  "chunks_created": 42,
  "vectors_stored": 42,
  "message": "Successfully processed and indexed document."
}
```

---

#### 3. 🔍 Hybrid Search Endpoint (`POST /api/v1/retrieval/search`)
Executes hybrid search combining dense cosine vectors and sparse BM25-like lexical weights, merges candidate lists using Reciprocal Rank Fusion (RRF), and applies GPU cross-encoder reranking (`BAAI/bge-reranker-v2-m3`).

**Request Body (`application/json`):**
```json
{
  "query": "How to handle acute anaphylaxis emergency?"
}
```

**Response (`200 OK`):**
```json
{
  "query": "How to handle acute anaphylaxis emergency?",
  "results": [
    {
      "text": "Administer intramuscular epinephrine (0.3mg 1:1000) immediately into lateral thigh...",
      "score": 0.92,
      "percentage_score": 92.0,
      "document_id": "doc_987",
      "source": "Anaphylaxis_Protocol.pdf",
      "pdf_page": 12,
      "document_page": 12,
      "section": "Emergency Interventions",
      "recommendation_id": "REC-ANAPH-02",
      "is_table": false
    }
  ]
}
```

---

#### 4. ⚖️ Standalone Rerank Endpoint (`POST /api/v1/retrieval/rerank`)
Exposes cross-encoder reranking directly to grade arbitrary query-document pairs.

**Request Body (`application/json`):**
```json
{
  "query": "treatment for severe hypothermia",
  "documents": [
    "Apply warm dry blankets and transport to medical facility immediately.",
    "Give aspirin for headache."
  ]
}
```

**Response (`200 OK`):**
```json
{
  "query": "treatment for severe hypothermia",
  "scores": [0.942, 0.012],
  "results": [
    { "index": 0, "score": 0.942, "text": "Apply warm dry blankets..." },
    { "index": 1, "score": 0.012, "text": "Give aspirin..." }
  ]
}
```

---

#### 5. 🟢 Health Check Endpoint (`GET /health`)
Verifies service availability.

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "app_name": "clinical-rag"
}
```

---

## 🏗️ Architecture & Pipeline Flow

```text
=================== INGESTION PIPELINE ===================
Uploaded PDF ──► Docling Parser ──► Furniture Cleaner ──► BGE Tokenizer Chunker
                                                                 │
                                                                 ▼
                                                     Colab BGE-M3 Embedding API
                                                    (Dense 1024d + Sparse Lexical)
                                                                 │
                                                                 ▼
                                                         Qdrant Vector DB
                                                 (Named Vectors: dense + sparse)

=================== HYBRID RETRIEVAL & RERANKING ===================
User Query ──► BGE-M3 Embedder ──► Qdrant Hybrid Search (Dense Top-20 & Sparse Top-20)
                                                 │
                                                 ▼
                                     Reciprocal Rank Fusion (RRF)
                                                 │
                                                 ▼
                                     GPU BGE Cross-Encoder Reranker
                                                 │
                                                 ▼
                                   >=80% Score Threshold Filter
                                                 │
                                                 ▼
                                 Gemini LLM (Scope & Knowledge Check)
                                                 │
                                                 ▼
                                   Bullet Answer + Exact Citations
```

---

## 🧪 Testing

Execute automated unit tests with `pytest`:

```bash
python -m pytest
```
