"""
pipeline/retriever.py
---------------------
Hybrid retrieval: BM25 + Dense (ChromaDB) → Cross-Encoder Re-ranking.

Pipeline
────────
1. Build BM25 index from all ChromaDB documents at startup (singleton)
2. For each sub-query:
   a. BM25 retrieval  → top TOP_K_BM25 candidates
   b. Dense retrieval → top TOP_K_DENSE candidates
   c. Union both sets
3. Cross-Encoder re-ranking over ~20 candidates → top TOP_K_RERANK
4. (Optional) Query reformulation: LLM rephrases query, repeat step 2-3

Re-ranking
──────────
We use  cross-encoder/ms-marco-MiniLM-L-6-v2 which scores (query, doc) pairs
directly, giving more accurate relevance than bi-encoder cosine similarity.

Query Reformulation
───────────────────
Triggered by the RETRIEVE macro-action sub-heuristic when:
  - Previous retrieve returned < 2 results, OR
  - Step > 1 and no new information was found

Strategy: ask Claude to rephrase to maximise legal retrieval diversity.
Fallback: keyword expansion via legal synonyms.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from core.config import Config
from rl.actions import RetrieveResult

log = logging.getLogger(__name__)

# ── Legal synonym map for keyword expansion fallback ──────────────────────────
LEGAL_SYNONYMS: Dict[str, List[str]] = {
    "right":        ["liberty", "freedom", "entitlement", "privilege"],
    "court":        ["tribunal", "bench", "judiciary", "adjudicator"],
    "law":          ["statute", "provision", "act", "legislation", "rule"],
    "parliament":   ["legislature", "house of parliament", "lok sabha", "rajya sabha"],
    "article":      ["provision", "section", "clause"],
    "citizenship":  ["nationality", "domicile"],
    "government":   ["state", "authority", "executive", "administration"],
    "president":    ["head of state", "union executive"],
    "power":        ["authority", "jurisdiction", "competence"],
    "fundamental":  ["basic", "constitutional", "inalienable"],
}

# ── Singletons ────────────────────────────────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None
_cross_enc:   Optional[CrossEncoder]        = None
_bm25_index:  Optional[object]              = None       # BM25Okapi
_bm25_docs:   Optional[List[Dict]]          = None       # [{id, text, metadata}]
_collection:  Optional[object]              = None       # chromadb.Collection


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _embed_model


def _get_cross_enc() -> CrossEncoder:
    global _cross_enc
    if _cross_enc is None:
        log.info("Loading cross-encoder: %s", Config.RERANK_MODEL)
        _cross_enc = CrossEncoder(Config.RERANK_MODEL, max_length=512)
    return _cross_enc


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.config import Settings
        import os
        os.makedirs(Config.DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(
            path=Config.DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _build_bm25_index() -> Tuple[object, List[Dict]]:
    """Build BM25 index from all ChromaDB documents. Called once."""
    global _bm25_index, _bm25_docs
    if _bm25_index is not None:
        return _bm25_index, _bm25_docs

    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        from nltk.tokenize import word_tokenize
        tokenize_fn = lambda t: word_tokenize(t.lower())
    except Exception:
        tokenize_fn = lambda t: t.lower().split()

    from rank_bm25 import BM25Okapi

    collection = _get_collection()
    count      = collection.count()
    if count == 0:
        log.warning("ChromaDB collection is empty — BM25 index will be empty.")
        _bm25_docs  = []
        _bm25_index = BM25Okapi([[]])
        return _bm25_index, _bm25_docs

    log.info("Building BM25 index from %d documents …", count)
    result = collection.get(
        include=["documents", "metadatas"],
        limit=count,
    )

    docs_raw: List[Dict] = []
    for i, doc_id in enumerate(result["ids"]):
        docs_raw.append({
            "id":       doc_id,
            "text":     result["documents"][i],
            "metadata": result["metadatas"][i] if result["metadatas"] else {},
        })

    tokenized = [tokenize_fn(d["text"]) for d in docs_raw]
    _bm25_index = BM25Okapi(tokenized, k1=Config.BM25_K1, b=Config.BM25_B)
    _bm25_docs  = docs_raw
    log.info("BM25 index built: %d docs", len(_bm25_docs))
    return _bm25_index, _bm25_docs


# ── BM25 Retrieval ────────────────────────────────────────────────────────────

def _bm25_retrieve(query: str) -> List[Dict]:
    """Return top-K BM25 candidate dicts."""
    bm25, docs = _build_bm25_index()
    if not docs:
        return []
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(query.lower())
    except Exception:
        tokens = query.lower().split()

    scores    = bm25.get_scores(tokens)
    top_idxs  = np.argsort(scores)[::-1][: Config.TOP_K_BM25]
    return [docs[i] | {"bm25_score": float(scores[i])} for i in top_idxs if scores[i] > 0]


# ── Dense Retrieval ───────────────────────────────────────────────────────────

def _dense_retrieve(query: str) -> List[Dict]:
    """Return top-K ChromaDB (dense) candidate dicts."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    model    = _get_embed_model()
    q_vec    = model.encode(query, normalize_embeddings=True).tolist()
    results  = collection.query(
        query_embeddings=[q_vec],
        n_results=min(Config.TOP_K_DENSE, collection.count()),
        include=["documents", "metadatas", "distances", "ids"],
    )

    candidates = []
    for i, doc_id in enumerate(results["ids"][0]):
        candidates.append({
            "id":            doc_id,
            "text":          results["documents"][0][i],
            "metadata":      results["metadatas"][0][i] if results["metadatas"] else {},
            "dense_score":   1.0 - float(results["distances"][0][i]),  # cosine→similarity
        })
    return candidates


# ── Cross-Encoder Re-ranking ──────────────────────────────────────────────────

def _rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Re-rank candidates with the cross-encoder and return top-K.

    Cross-encoder scores (query, doc) pairs directly for relevance —
    more accurate than bi-encoder cosine because it sees both texts jointly.
    """
    if not candidates:
        return []

    cross_enc = _get_cross_enc()
    pairs     = [(query, c["text"][:500]) for c in candidates]   # truncate for speed
    scores    = cross_enc.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_k  = ranked[: Config.TOP_K_RERANK]

    return [c | {"rerank_score": float(s)} for c, s in top_k]


# ── Query Reformulation ───────────────────────────────────────────────────────

def _keyword_expand(query: str) -> str:
    """Append legal synonyms for key words found in the query."""
    q_lower  = query.lower()
    additions: List[str] = []
    for word, synonyms in LEGAL_SYNONYMS.items():
        if word in q_lower:
            additions.extend(synonyms[:2])
    if additions:
        return query + " " + " ".join(additions)
    return query


def _llm_reformulate(query: str) -> str:
    """Ask Claude to rephrase the query for better retrieval diversity."""
    try:
        import anthropic
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("No Anthropic API key")
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        prompt = (
            f"Rephrase the following legal sub-question to maximise "
            f"retrieval diversity while preserving the meaning. "
            f"Return ONLY the rephrased question, no explanation.\n\n"
            f"Question: {query}"
        )
        message = client.messages.create(
            model=Config.LLM_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        log.warning("LLM reformulation failed (%s); using keyword expansion.", e)
        return _keyword_expand(query)


def reformulate_query(query: str) -> str:
    """Reformulate a retrieval query. Tries LLM first, then keyword expansion."""
    return _llm_reformulate(query)


# ── Main Retrieval Function ───────────────────────────────────────────────────

def retrieve(
    sub_questions: List[str],
    reformulate: bool = False,
) -> RetrieveResult:
    """
    Hybrid retrieval for a list of sub-questions.

    Steps
    ─────
    1. Optionally reformulate each sub-question
    2. For each sub-question: BM25 + dense → union → cross-encoder re-rank
    3. Deduplicate by doc id, keep top TOP_K_RERANK overall

    Returns RetrieveResult with .chunks and .texts attributes.
    """
    reformulated_queries: List[str] = []
    all_candidates: Dict[str, Dict] = {}   # id → doc dict

    queries = sub_questions if sub_questions else ["legal information"]

    for q in queries:
        active_q = q
        if reformulate:
            rq = reformulate_query(q)
            if rq != q:
                reformulated_queries.append(rq)
                active_q = rq

        bm25_cands  = _bm25_retrieve(active_q)
        dense_cands = _dense_retrieve(active_q)

        # Union: merge by id (keep richer dict)
        for c in bm25_cands + dense_cands:
            cid = c["id"]
            if cid not in all_candidates:
                all_candidates[cid] = c
            else:
                all_candidates[cid].update(c)

    # Re-rank the union with the combined / first query
    union_list   = list(all_candidates.values())
    rerank_query = reformulated_queries[0] if reformulated_queries else queries[0]
    ranked       = _rerank(rerank_query, union_list)

    return RetrieveResult(
        chunks               = ranked,
        was_reformulated     = len(reformulated_queries) > 0,
        reformulated_queries = reformulated_queries,
    )
