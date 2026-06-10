"""
backend/rag/chromadb_store.py
Manages ChromaDB vector store for chunk storage and retrieval.
"""
import logging
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from .embeddings import embed_texts

logger = logging.getLogger(__name__)


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def store_chunks(chunks: list[dict], dataset_name: str) -> int:
    """
    Store a list of chunks into ChromaDB under a namespaced collection.

    Each chunk must have: id, text, source, type
    Returns number of chunks stored.
    """
    client = _get_client()

    # Use dataset-scoped collection name (sanitize)
    collection_name = f"{CHROMA_COLLECTION}_{dataset_name[:40].replace(' ', '_').replace('.', '_')}"

    # Delete existing collection if present (re-index on upload)
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "source": c.get("source", ""),
            "type": c.get("type", ""),
            "column": c.get("column", ""),
            "dataset": dataset_name,
        }
        for c in chunks
    ]

    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
    return len(chunks)


def get_collection_name(dataset_name: str) -> str:
    return f"{CHROMA_COLLECTION}_{dataset_name[:40].replace(' ', '_').replace('.', '_')}"


def collection_exists(dataset_name: str) -> bool:
    client = _get_client()
    name = get_collection_name(dataset_name)
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False