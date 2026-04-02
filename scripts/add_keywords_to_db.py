import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.config import Settings
from core.config import Config
from ingestion.keyword_extractor import extract_keywords_batch
from ingestion.logger import get_logger

log = get_logger("add_keywords")


def parse_args():
    p = argparse.ArgumentParser(description="Add keyword metadata to ChromaDB chunks")
    p.add_argument("--preview", action="store_true", help="Show keywords without saving")
    p.add_argument("--limit",   type=int, default=None, help="Only process first N chunks (for testing)")
    return p.parse_args()


def get_collection():
    client = chromadb.PersistentClient(
        path=Config.DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    log.info("Connected to collection '%s' — %d chunks", Config.COLLECTION_NAME, collection.count())
    return collection


def fetch_all_chunks(collection, limit=None):
    total  = collection.count()
    target = min(total, limit) if limit else total
    log.info("Fetching %d chunks...", target)

    result = collection.get(
        limit=target,
        include=["documents", "metadatas"]
    )

    chunks = []
    for i in range(len(result["ids"])):
        chunks.append({
            "id":       result["ids"][i],
            "text":     result["documents"][i],
            "metadata": result["metadatas"][i],
        })

    log.info("Fetched %d chunks.", len(chunks))
    return chunks


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Adding keywords to ChromaDB chunks")
    print("=" * 60)

    collection = get_collection()

    if collection.count() == 0:
        print("  ERROR: DB is empty. Run ingest_pipeline.py first.")
        sys.exit(1)

    # Step 1 — Fetch
    print("\n  Step 1/3 — Fetching chunks from DB...")
    chunks = fetch_all_chunks(collection, limit=args.limit)

    # Step 2 — Extract keywords
    print(f"\n  Step 2/3 — Extracting keywords using KeyBERT...")
    print("  (First run loads the model — may take a moment)")
    keyword_list = extract_keywords_batch(chunks)

    # Step 3 — Preview or save
    if args.preview:
        print("\n  Preview (first 10 chunks):")
        for chunk, kw in zip(chunks[:10], keyword_list[:10]):
            art   = chunk["metadata"].get("article_num", chunk["id"])
            title = chunk["metadata"].get("title", "")[:40]
            print(f"    Article {art:>6}  |  {title:<40}  |  {kw}")
        print(f"\n  Would update {len(chunks)} chunks. Run without --preview to save.")
        return

    print(f"\n  Step 3/3 — Saving keywords to DB...")
    batch_size = 100
    ids       = [c["id"] for c in chunks]
    metadatas = [{**c["metadata"], "keywords": kw} for c, kw in zip(chunks, keyword_list)]

    for i in range(0, len(ids), batch_size):
        collection.update(
            ids       = ids[i:i + batch_size],
            metadatas = metadatas[i:i + batch_size],
        )
        log.info("  Updated %d / %d", min(i + batch_size, len(ids)), len(ids))

    # Verify
    sample = collection.get(ids=[chunks[0]["id"]], include=["metadatas"])
    saved_kw = sample["metadatas"][0].get("keywords", "MISSING")
    print(f"\n  Verification — Article {chunks[0]['metadata'].get('article_num')}: {saved_kw}")

    print("\n" + "=" * 60)
    print(f"  Done! {len(chunks)} chunks updated with keywords.")
    print("=" * 60)


if __name__ == "__main__":
    main()