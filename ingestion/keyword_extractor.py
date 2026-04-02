import re
from typing import List, Dict
from ingestion.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_N_KEYWORDS  = 10
MAX_NGRAM_SIZE  = 2      # unigrams + bigrams
DEDUP_THRESHOLD = 0.9    
MIN_KEYWORD_LEN = 3

# Legal stopwords — too generic to be useful in BM25
LEGAL_STOPWORDS = {
    "shall", "may", "said", "such", "any", "every", "all",
    "provided", "notwithstanding", "subject", "referred", "made",
    "accordance", "pursuant", "herein", "thereof", "thereto",
    "hereby", "whereas", "article", "clause", "section", "sub",
    "act", "law", "order", "rule", "regulation", "constitution",
    "parliament", "president", "government", "india", "union",
}

# ── YAKE extractor ────────────────────────────────────────────────────────────

_yake_extractor = None

def _get_yake():
    global _yake_extractor
    if _yake_extractor is None:
        try:
            import yake
            _yake_extractor = yake.KeywordExtractor(
                lan="en",
                n=MAX_NGRAM_SIZE,
                dedupLim=DEDUP_THRESHOLD,
                top=TOP_N_KEYWORDS + 5,   # fetch extra, filter below
                features=None,
            )
            log.info("YAKE extractor ready.")
        except ImportError:
            raise ImportError(
                "YAKE not installed. Run: pip install yake"
            )
    return _yake_extractor


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean(keywords: List[str]) -> List[str]:
    seen   = set()
    result = []
    for kw in keywords:
        kw = kw.lower().strip()
        kw = re.sub(r'[^a-z0-9 ]', '', kw).strip()   # remove special chars

        if len(kw) < MIN_KEYWORD_LEN:
            continue
        if kw in LEGAL_STOPWORDS:
            continue
        # Skip if all words in phrase are stopwords
        if all(w in LEGAL_STOPWORDS for w in kw.split()):
            continue
        if kw not in seen:
            seen.add(kw)
            result.append(kw)

    return result[:TOP_N_KEYWORDS]


# ── Public API ────────────────────────────────────────────────────────────────

def extract_keywords(text: str) -> str:
    """
    Extract keywords from a single article text using YAKE.

    Returns:
        Comma-separated keyword string for ChromaDB metadata.
        e.g. "fundamental rights, life, liberty, personal liberty, citizen"
        Returns empty string for very short chunks.
    """
    # Skip chunks that are too short to extract meaningful keywords
    if len(text.strip()) < 50:
        return ""

    extractor = _get_yake()

    # YAKE returns (keyword, score) — lower score = more important
    results  = extractor.extract_keywords(text)
    keywords = [kw for kw, score in results]
    cleaned  = _clean(keywords)

    return ", ".join(cleaned)


def extract_keywords_batch(chunks: List[Dict]) -> List[str]:
    """
    Extract keywords for all chunks.

    Args:
        chunks: list of dicts with "text" key

    Returns:
        List of comma-separated keyword strings (one per chunk).
    """
    log.info("Starting YAKE keyword extraction for %d chunks...", len(chunks))

    # Load extractor once
    _get_yake()

    results = []
    for i, chunk in enumerate(chunks):
        kw_string = extract_keywords(chunk["text"])
        results.append(kw_string)
        if (i + 1) % 100 == 0:
            log.info("  Keywords extracted: %d / %d", i + 1, len(chunks))

    log.info("Keyword extraction complete.")
    return results