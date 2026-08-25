import uuid
import logging
from typing import List, Tuple
from qdrant_client import QdrantClient, models

from interfaces.vector_store import VectorStore
from schemas.documents import DocumentChunk, EmbeddingResult
from config.settings import settings

logger = logging.getLogger(__name__)


class QdrantProvider(VectorStore):
    """Qdrant Vector Store Provider supporting named vectors (dense + sparse)."""

    def __init__(
        self,
        url: str = settings.QDRANT_URL,
        api_key: str = settings.QDRANT_API_KEY,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.url = url
        self.api_key = api_key or None
        self.collection_name = collection_name
        self.dimension = dimension
        self._client = None

    @property
    def client(self) -> QdrantClient:
        """Lazy load Qdrant client connection."""
        if self._client is None:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def ensure_collection(self) -> None:
        """Create Qdrant collection with named dense and sparse vectors if not existing."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with named vectors.")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.dimension,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams()
                    },
                )
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection '{self.collection_name}': {e}")
            raise RuntimeError(f"Qdrant connection/collection error: {e}")

    def document_exists(self, document_id: str) -> bool:
        """Check if points belonging to document_id (file hash) already exist in payload."""
        try:
            res, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=1,
            )
            return len(res) > 0
        except Exception:
            return False

    def upsert_points(self, chunks: List[DocumentChunk], embeddings: List[EmbeddingResult]) -> int:
        """Format chunks and embeddings into Qdrant PointStruct objects and upsert."""
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings lists must be non-empty and of equal length.")

        self.ensure_collection()

        points: List[models.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())

            # Prepare payload containing all metadata fields plus chunk text
            payload = chunk.metadata.model_dump()
            payload["text"] = chunk.text

            named_vectors = {
                "dense": emb.dense,
                "sparse": models.SparseVector(
                    indices=emb.sparse_indices,
                    values=emb.sparse_values,
                ),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=named_vectors,
                    payload=payload,
                )
            )

        # Batch upsert points
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Successfully upserted {len(points)} points into Qdrant collection '{self.collection_name}'.")
        return len(points)

    def upsert_document_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Upsert DocumentChunk objects containing embedded dense_vector, sparse_indices, sparse_values directly."""
        if not chunks:
            return 0

        self.ensure_collection()

        points: List[models.PointStruct] = []
        for chunk in chunks:
            point_id = str(uuid.uuid4())
            payload = chunk.metadata.model_dump()
            payload["text"] = chunk.text

            named_vectors = {
                "dense": chunk.dense_vector or [],
                "sparse": models.SparseVector(
                    indices=chunk.sparse_indices or [],
                    values=chunk.sparse_values or [],
                ),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=named_vectors,
                    payload=payload,
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(f"Successfully upserted {len(points)} DocumentChunk points into Qdrant collection '{self.collection_name}'.")
        return len(points)

    def _search_named_vector(self, name: str, vector_payload: any, limit: int) -> List[dict]:
        """Search named vector with fallback to REST HTTP endpoint if server version is < 1.10.0."""
        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=vector_payload,
                using=name,
                limit=limit,
                with_payload=True,
            )
            points = res.points if hasattr(res, "points") else res
            hits: List[dict] = []
            for p in points:
                hits.append({
                    "id": str(p.id),
                    "score": float(p.score),
                    "payload": dict(p.payload or {}),
                })
            return hits
        except Exception as e:
            logger.warning(f"qdrant_client query_points failed for '{name}' ({e}). Attempting REST HTTP /points/search fallback...")
            try:
                import httpx
                url = f"{self.url}/collections/{self.collection_name}/points/search"
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["api-key"] = self.api_key

                if name == "dense":
                    req_vector = {"name": "dense", "vector": vector_payload}
                else:
                    indices = vector_payload.indices if hasattr(vector_payload, "indices") else vector_payload.get("indices", [])
                    values = vector_payload.values if hasattr(vector_payload, "values") else vector_payload.get("values", [])
                    req_vector = {
                        "name": "sparse",
                        "vector": {
                            "indices": indices,
                            "values": values,
                        },
                    }

                body = {
                    "vector": req_vector,
                    "limit": limit,
                    "with_payload": True,
                }

                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(url, json=body, headers=headers)
                    if resp.status_code != 200:
                        raise RuntimeError(f"REST search failed with status HTTP {resp.status_code}: {resp.text}")
                    data = resp.json().get("result", [])
                    hits = []
                    for item in data:
                        hits.append({
                            "id": str(item.get("id")),
                            "score": float(item.get("score", 0.0)),
                            "payload": dict(item.get("payload") or {}),
                        })
                    return hits
            except Exception as rest_err:
                logger.error(f"Both query_points and REST /points/search failed for vector '{name}': {rest_err}")
                raise RuntimeError(f"{name.capitalize()} vector search failed: {rest_err}")

    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        dense_top_k: int = settings.DENSE_TOP_K,
        sparse_top_k: int = settings.SPARSE_TOP_K,
    ) -> Tuple[List[dict], List[dict]]:
        """Perform hybrid search against named vectors 'dense' and 'sparse' in Qdrant.
        Returns a tuple of (dense_hits, sparse_hits).
        """
        self.ensure_collection()

        dense_hits = self._search_named_vector("dense", dense_vector, dense_top_k)

        sparse_hits: List[dict] = []
        if sparse_indices and sparse_values:
            sparse_vector_obj = models.SparseVector(indices=sparse_indices, values=sparse_values)
            sparse_hits = self._search_named_vector("sparse", sparse_vector_obj, sparse_top_k)

        logger.info(f"Qdrant hybrid search retrieved {len(dense_hits)} dense hits and {len(sparse_hits)} sparse hits.")
        return dense_hits, sparse_hits


