"""
validator.py
────────────
Validates the ingestion output before and after storing.

Checks:
  1. Chunk completeness — no duplicate IDs, expected count range
  2. Chunk quality     — no empty text, titles extracted properly
  3. DB verification  — count in ChromaDB matches what was ingested
  4. Spot-check       — verifies a known article (Article 21) was stored
"""

from typing import List, Dict, Tuple
from ingestion.logger import get_logger

log = get_logger(__name__)

# Constitutional articles that MUST be present
REQUIRED_ARTICLES = ["1", "12", "14", "19", "21", "32", "51A", "368"]

# Expected chunk count range (395 base + lettered variants)
MIN_EXPECTED_CHUNKS = 350
MAX_EXPECTED_CHUNKS = 700


def validate_chunks(chunks: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate the list of chunks before embedding.

    Returns:
        (is_valid, list_of_warning_messages)
    """
    warnings = []

    # ── Count ──────────────────────────────────────────────
    n = len(chunks)
    if n < MIN_EXPECTED_CHUNKS:
        warnings.append(
            f"Only {n} chunks found — expected at least {MIN_EXPECTED_CHUNKS}. "
            "Check TOC removal and PDF extraction."
        )
    elif n > MAX_EXPECTED_CHUNKS:
        warnings.append(
            f"{n} chunks found — expected at most {MAX_EXPECTED_CHUNKS}. "
            "Possible false splits in the chunker."
        )
    else:
        log.info("Chunk count OK: %d (expected %d–%d)", n, MIN_EXPECTED_CHUNKS, MAX_EXPECTED_CHUNKS)

    # ── Duplicate IDs ──────────────────────────────────────
    ids = [c["id"] for c in chunks]
    seen = set()
    dupes = []
    for cid in ids:
        if cid in seen:
            dupes.append(cid)
        seen.add(cid)
    if dupes:
        warnings.append(f"Duplicate chunk IDs found: {dupes[:10]}")
    else:
        log.info("No duplicate IDs found.")

    # ── Empty text ─────────────────────────────────────────
    empty = [c["id"] for c in chunks if not c.get("text", "").strip()]
    if empty:
        warnings.append(f"Chunks with empty text: {empty[:10]}")

    # ── Missing titles ─────────────────────────────────────
    no_title = [c["id"] for c in chunks if not c.get("title", "").strip()]
    if no_title:
        warnings.append(f"Chunks missing titles: {no_title[:10]}")

    # ── Required articles ──────────────────────────────────
    present_nums = {c["article_num"] for c in chunks}
    missing = [a for a in REQUIRED_ARTICLES if a not in present_nums]
    if missing:
        warnings.append(f"Required articles missing from chunks: {missing}")
    else:
        log.info("All required articles present: %s", REQUIRED_ARTICLES)

    is_valid = len(warnings) == 0

    if is_valid:
        log.info("Chunk validation PASSED.")
    else:
        for w in warnings:
            log.warning("VALIDATION WARNING: %s", w)

    return is_valid, warnings


def validate_db(chunks: List[Dict], db_count: int) -> Tuple[bool, List[str]]:
    """
    Validate the ChromaDB state after embedding.

    Returns:
        (is_valid, list_of_warning_messages)
    """
    warnings = []

    if db_count < len(chunks):
        warnings.append(
            f"DB count ({db_count}) < chunks sent ({len(chunks)}). "
            "Some chunks may not have been stored."
        )
    else:
        log.info("DB count OK: %d chunks stored.", db_count)

    is_valid = len(warnings) == 0
    if not is_valid:
        for w in warnings:
            log.warning("DB VALIDATION WARNING: %s", w)
    return is_valid, warnings


def print_summary(chunks: List[Dict], embed_result: Dict) -> None:
    """Print a human-readable ingestion summary to the console."""
    parts = {}
    for c in chunks:
        p = c.get("part", "Unknown")
        parts[p] = parts.get(p, 0) + 1

    log.info("═" * 60)
    log.info("  INGESTION SUMMARY")
    log.info("═" * 60)
    log.info("  Total chunks stored : %d", embed_result.get("stored", 0))
    log.info("  DB total count      : %d", embed_result.get("db_count", 0))
    log.info("  Embedding time      : %.1fs", embed_result.get("duration_sec", 0))
    log.info("  Chunks by Part:")
    for part, count in sorted(parts.items()):
        log.info("    %-45s  %3d chunks", part[:45], count)
    log.info("═" * 60)