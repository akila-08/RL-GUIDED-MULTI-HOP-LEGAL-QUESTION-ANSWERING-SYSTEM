from __future__ import annotations

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from core.config import Config

# ---------------------------------------------------------------------------
# Singleton embedding model (shared with retriever to avoid double-loading)
# ---------------------------------------------------------------------------
_embed_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _embed_model


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts; return (N, 384) normalised float32 array."""
    if not texts:
        return np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype(np.float32)


def _mean_pool(texts: List[str]) -> np.ndarray:
    """Return mean-pooled embedding (384,) or zeros if texts is empty."""
    if not texts:
        return np.zeros(Config.EMBEDDING_DIM, dtype=np.float32)
    vecs = _embed(texts)           # (N, 384)
    return vecs.mean(axis=0)       # (384,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_state(
    question: str,
    sub_questions: List[str],
    doc_texts: List[str],
    sub_answers: List[str],
    complexity_score: float,
    step: int,
    max_steps: int,
) -> np.ndarray:
    """
    Construct the state vector s_t ∈ ℝ^1538.

    Parameters
    ----------
    question         : original user question
    sub_questions    : list of generated sub-questions so far
    doc_texts        : list of retrieved document chunk texts so far
    sub_answers      : list of generated sub-answers so far
    complexity_score : float in [0, 1] from the classifier
    step             : current step index (0-based)
    max_steps        : maximum allowed steps per episode

    Returns
    -------
    np.ndarray of shape (1538,)
    """
    e_q    = _embed([question])[0]          # (384,)
    e_subs = _mean_pool(sub_questions)      # (384,)
    e_docs = _mean_pool(doc_texts)          # (384,)
    e_ans  = _mean_pool(sub_answers)        # (384,)

    c_t = np.array([float(complexity_score)], dtype=np.float32)          # (1,)
    n_t = np.array([step / max(max_steps, 1)], dtype=np.float32)         # (1,)

    state = np.concatenate([e_q, e_subs, e_docs, e_ans, c_t, n_t])      # (1538,)
    return state
