"""
embedder.py
───────────
Embeds article chunks and stores them in ChromaDB.

Responsibilities:
  1. Load the sentence-transformer embedding model
  2. Embed chunks in configurable batches (avoids OOM on large sets)
  3. Upsert into ChromaDB — safe to re-run (won't duplicate)
  4. Verify stored count matches input count
"""

import time
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from core.config import Config
from ingestion.logger import get_logger

log = get_logger(__name__)


# ── Singleton model + collection ─────────────────────────────────────────────
# Loaded once, reused across calls (important for the API server).

_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", Config.EMBEDDING_MODEL)
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
        log.info("Embedding model loaded — dimension: %d", _model.get_sentence_embedding_dimension())
    return _model


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        import os
        os.makedirs(Config.DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(
            path=Config.DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        log.info(
            "ChromaDB collection '%s' ready — current count: %d",
            Config.COLLECTION_NAME, _collection.count()
        )
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def embed_and_store(chunks: List[Dict]) -> Dict:
    """
    Embed all chunks and upsert them into ChromaDB.

    Uses upsert (not add) so the pipeline is safely re-runnable.
    Chunks are processed in batches to control memory usage.

    Args:
        chunks: List of chunk dicts from chunker.chunk_by_article()

    Returns:
        Summary dict:  { "total": int, "stored": int, "duration_sec": float }
    """
    if not chunks:
        log.warning("embed_and_store called with empty chunk list.")
        return {"total": 0, "stored": 0, "duration_sec": 0.0}

    model      = get_embedding_model()
    collection = get_collection()
    batch_size = Config.EMBEDDING_BATCH_SZ

    t_start = time.perf_counter()
    total   = len(chunks)
    stored  = 0

    log.info("Starting embedding — %d chunks, batch_size=%d", total, batch_size)

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        ids        = [c["id"]          for c in batch]
        texts      = [c["text"]        for c in batch]
        metadatas  = [_build_metadata(c) for c in batch]

        # Embed
        embeddings = model.encode(
            texts,
            batch_size  = batch_size,
            show_progress_bar = False,
            normalize_embeddings = True,       # unit-length vectors for cosine sim
        ).tolist()

        # Upsert — overwrites if id already exists
        collection.upsert(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas,
        )

        stored += len(batch)
        log.info(
            "  Batch %d–%d upserted (%d/%d)",
            batch_start + 1, batch_start + len(batch), stored, total
        )

    duration = time.perf_counter() - t_start

    # Verify
    actual_count = collection.count()
    log.info(
        "Embedding complete in %.1fs | requested: %d | in DB now: %d",
        duration, total, actual_count
    )

    if actual_count < total:
        log.warning(
            "DB count (%d) is less than chunks sent (%d) — possible upsert issue.",
            actual_count, total
        )

    return {
        "total":        total,
        "stored":       stored,
        "db_count":     actual_count,
        "duration_sec": round(duration, 2),
    }


def collection_stats() -> Dict:
    """Return basic statistics about the current ChromaDB collection."""
    collection = get_collection()
    count = collection.count()

    stats = {"collection": Config.COLLECTION_NAME, "total_chunks": count}

    if count > 0:
        # Peek at a few records to show sample metadata
        sample = collection.peek(limit=3)
        stats["sample_ids"]    = sample.get("ids", [])
        stats["sample_titles"] = [
            m.get("title", "?") for m in sample.get("metadatas", [])
        ]

    return stats


def reset_collection() -> None:
    """
    Delete and recreate the collection. Useful for a clean re-ingest.
    WARNING: Destroys all stored data.
    """
    global _collection
    import os
    os.makedirs(Config.DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(
        path=Config.DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(Config.COLLECTION_NAME)
        log.warning("Collection '%s' deleted.", Config.COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet

    _collection = client.create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    log.info("Collection '%s' recreated (empty).", Config.COLLECTION_NAME)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_metadata(chunk: Dict) -> Dict:
    """
    Build the metadata dict stored alongside each chunk in ChromaDB.
    ChromaDB metadata values must be str, int, or float — no lists or None.
    """
    return {
        "article_num": chunk.get("article_num", ""),
        "title":       chunk.get("title",       "")[:500],   # cap at 500 chars
        "part":        chunk.get("part",         ""),
        "char_count":  chunk.get("char_count",   0),
    }