import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from Ai_strategy_engine.logger import logger


# Cached per worker process — list/describe only happens once
_INDEX_CACHE = None


def _get_index():
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    pc         = Pinecone(api_key=os.environ.get("pinecone_Api_key", ""))
    index_name = os.environ.get("PINECONE_INDEX_NAME", "documents")

    logger.info("Listing indexes")
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                region=os.environ.get("PINECONE_REGION", "us-east-1"),
            ),
        )
        logger.info(f"[Pinecone] Created index '{index_name}'")

    logger.info(f"Index client created for index '{index_name}'")
    _INDEX_CACHE = pc.Index(index_name)
    return _INDEX_CACHE


class PineconeService:

    def index_document(self, doc_id: str, doc_name: str, chunks_meta: list[dict]) -> int:
        """Upsert all chunks for a document. Returns number of vectors upserted.

        chunks_meta: list of {"text": str, "page": int}
        """
        from main_app.embedding_service import Embedding

        if not chunks_meta:
            return 0

        texts = [c["text"] for c in chunks_meta]
        emb = Embedding()
        embeddings = emb.create_embeddings(texts)

        vectors = [
            {
                "id": f"{doc_id}-{i}",
                "values": embeddings[i],
                "metadata": {
                    "doc_id":   doc_id,
                    "doc_name": doc_name,
                    "page":     chunks_meta[i].get("page", 0),
                    "text":     texts[i],
                },
            }
            for i in range(len(texts))
        ]

        index = _get_index()
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            index.upsert(vectors=vectors[start : start + batch_size])

        logger.info(f"[Pinecone] Upserted {len(vectors)} vectors for doc {doc_id}")
        return len(vectors)

    def query(self, doc_ids: list[str], query_vec: list[float], top_k: int = 10) -> list[dict]:
        """Semantic search filtered to selected doc_ids. Returns list of {text, score}."""
        if not doc_ids:
            return []

        index = _get_index()
        results = index.query(
            vector=query_vec,
            top_k=top_k,
            filter={"doc_id": {"$in": doc_ids}},
            include_metadata=True,
        )
        return [
            {
                "text":     m.metadata.get("text", ""),
                "score":    m.score,
                "doc_name": m.metadata.get("doc_name", ""),
                "page":     m.metadata.get("page", 0),
            }
            for m in results.matches
            if m.metadata
        ]

    def delete_document(self, doc_id: str):
        """Delete all vectors belonging to a document."""
        index = _get_index()
        index.delete(filter={"doc_id": doc_id})
        logger.info(f"[Pinecone] Deleted vectors for doc {doc_id}")
