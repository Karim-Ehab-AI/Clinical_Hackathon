# Clinical RAG MVP — Phase 1 & Phase 2

A modular, SOLID-compliant FastAPI service for clinical document ingestion, hybrid retrieval (dense + sparse), Reciprocal Rank Fusion (RRF), cross-encoder reranking, and a modern React Frontend UI.

---

## 🚀 Getting Started / طريقة تشغيل المشروع

Follow these steps in order to start the full system:

### 📋 Prerequisites
1. **Docker Desktop** installed and running.
2. **Python 3.10+** (Conda environment `clinical-rag` recommended).
3. **Node.js 18+** & `npm`.

---

### 1️⃣ Step 1: Open Docker Desktop & Start Vector DB
First, launch **Docker Desktop**. Once Docker is active, run the following command from the project root directory (`d:\Clinical_Hackathon`) to start the Qdrant container:

```bash
docker compose up -d
```
> 📍 **Qdrant Vector DB:** Running at `http://localhost:6333`

---

### 2️⃣ Step 2: Run Google Colab GPU Service & Start FastAPI Backend

1. **Run Google Colab Notebook:**
   - Open and run the Google Colab GPU notebook (for `BAAI/bge-m3` embedding and `bge-reranker-v2-m3` reranking services).
   - Copy the generated `ngrok` public URL (e.g., `https://xxxx.ngrok-free.app`).
   - Update your `.env` file in the root directory with the ngrok URL:
     ```ini
     EMBEDDING_API_URL=https://your-ngrok-domain.ngrok-free.app
     RERANKER_API_URL=https://your-ngrok-domain.ngrok-free.app
     ```

2. **Start the FastAPI Backend:**
   Open a terminal, activate your environment, navigate to the `src` folder, and start Uvicorn:

```bash
# Activate Conda environment
conda activate clinical-rag

# Navigate to backend directory and start server
cd src
uvicorn main:app --host 0.0.0.0 --port 3000 --reload
```
> 📍 **Backend API:** `http://localhost:3000`  
> 📍 **Swagger Docs:** `http://localhost:3000/docs`

---

### 3️⃣ Step 3: Start the Frontend UI
Open a **new terminal**, navigate to `src/UI`, install dependencies (if not done previously), and start Vite:

```bash
# Navigate to UI directory
cd src/UI

# Install dependencies (first time only)
npm install

# Run Vite dev server
npm run dev
```
> 📍 **Frontend UI:** `http://localhost:8080/`

---

## 🏗️ End-to-End Architecture Overview

```text
=================== PHASE 1: INGESTION PIPELINE ===================
Uploaded PDF ──► DoclingProvider ──► CleaningService ──► ChunkingService
                                                               │
                                                               ▼
                                                   ColabEmbeddingProvider
                                                   (Dense + Sparse BGE-M3 API)
                                                               │
                                                               ▼
                                                       QdrantProvider
                                               (Named Vectors: dense + sparse)

=================== PHASE 2: HYBRID RETRIEVAL & RERANKING ===================
Query (text)
    │
    ▼
ColabEmbeddingProvider.embed_query() ──► Dense Vector (1024) + Sparse Vector (indices/values)
    │
    ▼
QdrantProvider.hybrid_search()      ──► Top DENSE_TOP_K & Top SPARSE_TOP_K candidate lists
    │
    ▼
RetrievalService.RRF_fusion()       ──► Reciprocal Rank Fusion (RRF) ──► Top HYBRID_TOP_K candidates
    │
    ▼
RerankingService / ColabReranker    ──► BAAI/bge-reranker-v2-m3 (or NoOpReranker Fallback)
    │
    ▼
Top RERANK_TOP_K Ranked Results with Provenance Metadata (page, section, recommendation_id, is_table)
```

---

## ⚡ Configuration & Environment Variables

Update `.env` in the root directory with your configuration:

```ini
APP_NAME=clinical-rag
DEBUG=true

# Qdrant Settings
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=clinical_documents

# Remote Colab Embedding Service
EMBEDDING_API_URL=https://your-ngrok-domain.ngrok-free.app
EMBEDDING_API_KEY=
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_TIMEOUT=30
EMBEDDING_BATCH_SIZE=32

# Retrieval & Search Settings
DENSE_TOP_K=20
SPARSE_TOP_K=20
HYBRID_TOP_K=20
RERANK_TOP_K=10
RRF_K=60

# Remote Colab Cross-Encoder Reranker Service
RERANKER_ENABLED=true
RERANKER_API_URL=https://your-ngrok-domain.ngrok-free.app
RERANKER_API_KEY=
RERANKER_TIMEOUT=30
```

---

## 📡 API Endpoints

### 1. 🤖 Generation Endpoint (`POST /api/v1/generation/generate`)
Used by the frontend to obtain evidence-based clinical guidance:

```bash
curl -X POST "http://localhost:3000/api/v1/generation/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "What should I do for a severe burn on the arm?"}'
```

### 2. 🔍 Hybrid Search Endpoint (`POST /api/v1/retrieval/search`)

```bash
curl -X POST "http://localhost:3000/api/v1/retrieval/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What HbA1c target is recommended for adults with type 2 diabetes?"}'
```

---

## 🔄 Fusion & Reranking Strategy

### 1. Reciprocal Rank Fusion (RRF)
Raw similarity scores from dense cosine distance and sparse lexical weights operate on completely different scales. To combine candidate items without biased weighting, we apply **Reciprocal Rank Fusion**:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}$$

Where:
- $k = 60$ (configurable via `RRF_K`).
- $\text{rank}_m(d)$ is the 1-indexed position of document $d$ in result set $m$.
- RRF merges candidate items into the top `HYBRID_TOP_K` (20) documents.

### 2. Cross-Encoder Reranking (`ColabReranker`) & Graceful Fallback (`NoOpReranker`)
- Candidate items from RRF fusion are submitted to the GPU cross-encoder (`BAAI/bge-reranker-v2-m3`) in **one single batch call**.
- Scores returned by the reranker endpoint correspond strictly to candidate index order. Reranked documents are then explicitly sorted descending by the cross-encoder score before returning the top `RERANK_TOP_K` (10) items.
- If the remote reranker endpoint is disabled (`RERANKER_ENABLED=false`) or unreachable, `RerankingService` automatically degrades to `NoOpReranker`, logging a warning and preserving the fused RRF ranking without failing search requests.

---

## 🧪 Testing

Run the full automated pytest suite:
```bash
python -m pytest
```
