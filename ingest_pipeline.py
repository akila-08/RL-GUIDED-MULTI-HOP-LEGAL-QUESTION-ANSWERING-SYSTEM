#!/usr/bin/env python3
"""
ingest_pipeline.py
──────────────────
Top-level runner that executes the full ingestion pipeline.

RUN THIS ONCE before starting the API or any training scripts.

Usage:
    python ingest_pipeline.py              # normal ingestion
    python ingest_pipeline.py --reset      # wipe DB and re-ingest
    python ingest_pipeline.py --stats      # show DB stats and exit
    python ingest_pipeline.py --dry-run    # extract + chunk only, no embedding
"""

import sys
import os
import argparse

# Make sure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config   import Config
from ingestion     import (
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

log = get_logger("ingest_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Ingest Constitution of India PDF into ChromaDB")
    p.add_argument("--reset",   action="store_true", help="Delete existing collection and re-ingest from scratch")
    p.add_argument("--stats",   action="store_true", help="Print current DB stats and exit")
    p.add_argument("--dry-run", action="store_true", help="Run extract + chunk only, skip embedding")
    return p.parse_args()


def _banner(step, total, current):
    print(f"\n{'='*60}")
    print(f"  STEP {current}/{total}  --  {step}")
    print(f"{'='*60}")


def _print_stats():
    stats = collection_stats()
    print("\n  ChromaDB Stats")
    print(f"   Collection   : {stats['collection']}")
    print(f"   Total chunks : {stats['total_chunks']}")
    if stats["total_chunks"] > 0:
        print(f"   Sample IDs   : {stats.get('sample_ids', [])}")
        print(f"   Sample titles: {stats.get('sample_titles', [])}")


def _check_pdf():
    if not os.path.exists(Config.PDF_PATH):
        print(f"\n  ERROR: PDF not found at: {Config.PDF_PATH}")
        print( "    Place 'constitution_of_india.pdf' inside the data/ folder.")
        sys.exit(1)


def run(reset=False, dry_run=False):
    total_steps = 3 if dry_run else 4

    if reset:
        confirm = input("WARNING: This will DELETE all stored data. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        reset_collection()
        print("Collection wiped and recreated.")

    _check_pdf()

    _banner("Extracting text from PDF", total_steps, 1)
    body_text = extract_body_text(Config.PDF_PATH)
    print(f"  OK  Extracted {len(body_text):,} characters from {Config.PDF_PATH}")

    _banner("Chunking by article", total_steps, 2)
    chunks = chunk_by_article(body_text)
    print(f"  OK  Created {len(chunks)} article chunks")
    print("\n  Preview (first 5 chunks):")
    for c in chunks[:5]:
        print(f"    Article {c['article_num']:>6}  |  {c['title'][:55]:<55}  |  {c['char_count']} chars")

    _banner("Validating chunks", total_steps, 3)
    is_valid, warnings = validate_chunks(chunks)
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    - {w}")
    if not is_valid:
        proceed = input("\n  Validation issues found. Proceed anyway? [y/N]: ")
        if proceed.strip().lower() != "y":
            print("  Aborted.")
            sys.exit(1)
    else:
        print("  OK  All validation checks passed")

    if dry_run:
        print(f"\n  [Dry-run] Skipping embedding. {len(chunks)} chunks would be stored.")
        sys.exit(0)

    _banner("Embedding and storing in ChromaDB", total_steps, 4)
    print("  (First run downloads the model -- may take 1-3 minutes)")
    result = embed_and_store(chunks)

    db_valid, db_warnings = validate_db(chunks, result["db_count"])
    if db_warnings:
        for w in db_warnings:
            print(f"  WARNING: {w}")

    print_summary(chunks, result)

    print(f"\n{'='*60}")
    print( "  Ingestion complete!")
    print(f"  Chunks stored : {result['stored']}")
    print(f"  DB total count: {result['db_count']}")
    print(f"  DB location   : {Config.DB_PATH}")
    print(f"  Time taken    : {result['duration_sec']}s")
    print(f"{'='*60}")
    print("\n  Start the API server with:")
    print("  uvicorn api.main:app --reload\n")


if __name__ == "__main__":
    args = parse_args()
    if args.stats:
        _print_stats()
        sys.exit(0)
    run(reset=args.reset, dry_run=vars(args).get("dry_run", False))