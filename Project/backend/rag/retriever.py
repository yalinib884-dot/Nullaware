"""
backend/rag/retriever.py
Retrieves top-k relevant chunks from ChromaDB for a given query.
"""
import logging
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, RAG_TOP_K
from .embeddings import embed_query
from .chromadb_store import get_collection_name

logger = logging.getLogger(__name__)


def retrieve_chunks(query: str, dataset_name: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """
    Retrieve top-k semantically similar chunks for a query.

    Returns list of dicts: { text, source, type, column, distance }
    """
    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection_name = get_collection_name(dataset_name)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        logger.warning(f"Collection '{collection_name}' not found. Run indexing first.")
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results.get("documents"):
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "type": meta.get("type", ""),
                "column": meta.get("column", ""),
                "distance": round(float(dist), 4),
            })

    logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}...'")
    return chunks