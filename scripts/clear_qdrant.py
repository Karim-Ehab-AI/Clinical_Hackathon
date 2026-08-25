import sys
from pathlib import Path

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

import logging
from qdrant_client import QdrantClient, models
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_qdrant_collection():
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    collection_name = settings.QDRANT_COLLECTION_NAME

    logger.info(f"Connecting to Qdrant at '{settings.QDRANT_URL}'...")
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        logger.info(f"🗑️ Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"✅ Collection '{collection_name}' deleted successfully.")

    logger.info(f"✨ Re-creating fresh collection '{collection_name}' with named vectors...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )
    logger.info(f"🎉 Qdrant collection '{collection_name}' is now completely fresh and ready for indexing!")


if __name__ == "__main__":
    clear_qdrant_collection()
