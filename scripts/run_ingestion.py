#!/usr/bin/env python3
"""
run_ingestion.py
────────────────
One-time script to ingest the Constitution of India PDF into ChromaDB.

Usage:
    python scripts/run_ingestion.py              # normal run
    python scripts/run_ingestion.py --reset      # wipe DB and re-ingest
    python scripts/run_ingestion.py --stats      # just show DB stats
    python scripts/run_ingestion.py --validate   # validate existing DB only

Run this ONCE before starting the API server.
"""

import sys
import os
import argparse

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from ingestion   import (
    extract_body_text,
    chunk_by_article,
    embed_and_store,
    collection_stats,
    reset_collection,
    validate_chunks,
    validate_db,
    print_summary,
)
from ingestion.logger import get_logger

log = get_logger("run_ingestion")


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest Constitution PDF into ChromaDB")
    parser.add_argument("--reset",    action="store_true", help="Wipe the DB and re-ingest from scratch")
    parser.add_argument("--stats",    action="store_true", help="Print DB stats and exit")
    parser.add_argument("--validate", action="store_true", help="Validate the DB without re-ingesting")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stats:
        stats = collection_stats()
        print("\n📊 ChromaDB Stats")
        print(f"   Collection  : {stats['collection']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        if stats["total_chunks"] > 0:
            print(f"   Sample IDs  : {stats.get('sample_ids', [])}")
            print(f"   Sample titles: {stats.get('sample_titles', [])}")
        sys.exit(0)

    if args.validate:
        stats = collection_stats()
        n = stats["total_chunks"]
        if n == 0:
            print("❌  DB is empty. Run without --validate to ingest first.")
            sys.exit(1)
        print(f"✅  DB contains {n} chunks.")
        sys.exit(0)

    if args.reset:
        confirm = input("⚠️  This will DELETE all stored data. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        reset_collection()
        print("🗑️   Collection reset.")

    if not os.path.exists(Config.PDF_PATH):
        print(f"\n❌  PDF not found at: {Config.PDF_PATH}")
        print("    Please place 'constitution_of_india.pdf' in the data/ folder.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  STEP 1/4  —  Extracting text from PDF")
    print("=" * 60)
    body_text = extract_body_text(Config.PDF_PATH)
    print(f"  OK  Extracted {len(body_text):,} characters")

    print("\n" + "=" * 60)
    print("  STEP 2/4  —  Chunking by article")
    print("=" * 60)
    chunks = chunk_by_article(body_text)
    print(f"  OK  Created {len(chunks)} article chunks")

    print("\n" + "=" * 60)
    print("  STEP 3/4  —  Validating chunks")
    print("=" * 60)
    is_valid, warnings = validate_chunks(chunks)
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    if not is_valid:
        proceed = input("\n  Validation failed. Proceed anyway? [y/N]: ")
        if proceed.strip().lower() != "y":
            print("  Aborted.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  STEP 4/4  —  Embedding and storing in ChromaDB")
    print("=" * 60)
    print("  (First run may take 1-3 mins — model download + encoding)")
    result = embed_and_store(chunks)

    db_valid, db_warnings = validate_db(chunks, result["db_count"])
    if db_warnings:
        for w in db_warnings:
            print(f"  WARNING: {w}")

    print_summary(chunks, result)

    print("\n" + "=" * 60)
    print(f"  Ingestion complete!")
    print(f"  {result['stored']} chunks stored | Location: {Config.DB_PATH}")
    print(f"  Time: {result['duration_sec']}s")
    print("=" * 60)
    print("\n  Start the API with:")
    print("  $ uvicorn api.main:app --reload\n")


if __name__ == "__main__":
    main()