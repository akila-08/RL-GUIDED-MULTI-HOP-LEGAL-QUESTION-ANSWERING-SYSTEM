"""
rl/rewards.py
-------------
8 reward signal functions for the RL-Guided Legal QA system.

Signal      Direction  Description
──────────────────────────────────────────────────────────────
correctness   +pos      ROUGE-L of final answer vs gold answer
retrieval     +pos      cosine sim between sub-q and chunks
hallucination −neg      unsupported entity penalty
entity        +pos      article/entity coverage in retrieved docs
entailment    +pos      NLI entailment score (context → answer)
fluency       +pos      grammar / language quality score
conciseness   +pos      length efficiency
query_align   +pos      semantic similarity answer ↔ original question

All individual rewards are in [0, 1]. Combined reward:
  R = w_cor*cor + w_ret*ret - w_hal*hal + w_ent*ent
      + w_nli*nli + w_flu*flu + w_con*con + w_algn*algn
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import Config

log = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────────────────
_embed_model:    Optional[SentenceTransformer] = None
_nli_model:      Optional[object]              = None   # CrossEncoder
_spacy_nlp:      Optional[object]              = None


def _embed_model_instance() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _embed_model


def _nli_instance():
    """Cross-encoder for NLI — loaded once on first use."""
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def _spacy_instance():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            _spacy_nlp = None
    return _spacy_nlp


# ── Helper ────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _embed_text(text: str) -> np.ndarray:
    model = _embed_model_instance()
    return model.encode(text, normalize_embeddings=True).astype(np.float32)


def _embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)
    model = _embed_model_instance()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _extract_entities(text: str) -> List[str]:
    """Extract named entities using spaCy; fallback to capitalised words."""
    nlp = _spacy_instance()
    if nlp is not None:
        doc = nlp(text)
        return [ent.text.lower() for ent in doc.ents]
    # Fallback: match "Article X" patterns + capitalised tokens
    articles = re.findall(r'article\s+\d+[a-z]?(?:\([^)]*\))?', text.lower())
    caps     = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    return articles + [c.lower() for c in caps]


# ── Individual reward functions ───────────────────────────────────────────────

def reward_correctness(final_answer: str, gold_answer: Optional[str]) -> float:
    """
    ROUGE-L recall of final_answer against gold_answer.
    Returns 0.0 if no gold is available (RL training without labels).
    """
    if not gold_answer or not final_answer:
        return 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score  = scorer.score(gold_answer, final_answer)
        return float(score["rougeL"].recall)
    except Exception as e:
        log.warning("ROUGE computation failed: %s", e)
        return 0.0


def reward_retrieval(sub_questions: List[str], doc_texts: List[str]) -> float:
    """
    Mean cosine similarity between sub-question embeddings and
    the mean chunk embedding.  Measures how relevant retrieved docs are.
    """
    if not sub_questions or not doc_texts:
        return 0.0
    q_vecs   = _embed_texts(sub_questions)   # (n_q, 384)
    doc_vecs = _embed_texts(doc_texts)       # (n_d, 384)
    q_mean   = q_vecs.mean(axis=0)
    d_mean   = doc_vecs.mean(axis=0)
    sim      = _cosine(q_mean, d_mean)
    # Map cosine sim from [-1,1] to [0,1]
    return float(np.clip((sim + 1) / 2, 0.0, 1.0))


def reward_hallucination(final_answer: str, doc_texts: List[str]) -> float:
    """
    Negative penalty: fraction of entities in the answer that are NOT
    supported by any retrieved document.  Returns value in [0, 1].
    Higher → more hallucinated (used as PENALTY: subtract in combined reward).
    """
    if not final_answer or not doc_texts:
        return 0.0
    context_blob = " ".join(doc_texts).lower()
    entities     = _extract_entities(final_answer)
    if not entities:
        return 0.0
    unsupported = sum(1 for e in entities if e not in context_blob)
    return float(unsupported / len(entities))


def reward_entity(final_answer: str, doc_texts: List[str]) -> float:
    """
    Fraction of entities/article refs in the answer that appear in
    the retrieved context.  Rewards entity consistency.
    """
    if not final_answer or not doc_texts:
        return 0.0
    context_blob = " ".join(doc_texts).lower()
    entities     = _extract_entities(final_answer)
    if not entities:
        return 0.5   # no entities → neutral
    supported = sum(1 for e in entities if e in context_blob)
    return float(supported / len(entities))


def reward_entailment(final_answer: str, doc_texts: List[str]) -> float:
    """
    NLI entailment score: P(answer is entailed by context).
    Uses cross-encoder/nli-deberta-v3-small.
    Labels: contradiction=0, neutral=1, entailment=2
    """
    if not final_answer or not doc_texts:
        return 0.0
    try:
        cross_enc = _nli_instance()
        # Truncate context to keep within model max-length
        context   = " ".join(doc_texts)[:2000]
        scores    = cross_enc.predict([(context, final_answer)])
        # scores shape: (1, 3) for [contradiction, neutral, entailment]
        probs     = _softmax(scores[0])
        return float(probs[2])   # entailment class
    except Exception as e:
        log.warning("Entailment scoring failed: %s", e)
        return 0.0


def reward_fluency(final_answer: str) -> float:
    """
    Grammar / language quality proxy.
    Uses a simple heuristic:
      - Sentence structure check (ends with '.', contains subject)
      - Penalise very short answers (< 10 words)
      - Penalise excessive repetition
    Returns float in [0, 1].
    """
    if not final_answer or len(final_answer.strip()) < 5:
        return 0.0

    score = 1.0
    words = final_answer.split()
    n_words = len(words)

    # Too short
    if n_words < 10:
        score -= 0.3

    # Does not end with punctuation
    if final_answer.strip()[-1] not in ".!?":
        score -= 0.15

    # Repetition: fraction of duplicate bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    if bigrams:
        unique_ratio = len(set(bigrams)) / len(bigrams)
        if unique_ratio < 0.6:
            score -= 0.25

    # Prefer complete sentences (contains a verb hint)
    verb_patterns = ["is", "are", "was", "were", "provides", "states", "ensures",
                     "grants", "prohibits", "allows", "requires", "gives"]
    has_verb = any(v in final_answer.lower() for v in verb_patterns)
    if not has_verb:
        score -= 0.1

    return float(np.clip(score, 0.0, 1.0))


def reward_conciseness(final_answer: str) -> float:
    """
    Length efficiency: penalise answers that are much longer than needed.
    Score = 1 - (len / MAX_ANSWER_LEN), clipped to [0, 1].
    """
    if not final_answer:
        return 0.0
    ratio = len(final_answer) / max(Config.MAX_ANSWER_LEN, 1)
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def reward_query_alignment(question: str, final_answer: str) -> float:
    """
    Cosine similarity between the original question embedding and the
    final answer embedding.  Ensures the answer is on-topic.
    """
    if not question or not final_answer:
        return 0.0
    q_vec = _embed_text(question)
    a_vec = _embed_text(final_answer)
    sim   = _cosine(q_vec, a_vec)
    return float(np.clip((sim + 1) / 2, 0.0, 1.0))


# ── Combined reward ───────────────────────────────────────────────────────────

def compute_all_rewards(
    question: str,
    final_answer: str,
    sub_questions: List[str],
    doc_texts: List[str],
    gold_answer: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute all 8 rewards and return as a dict.
    Does NOT apply weights — weights are applied in the combined_reward() call.
    """
    return {
        "correctness":   reward_correctness(final_answer, gold_answer),
        "retrieval":     reward_retrieval(sub_questions, doc_texts),
        "hallucination": reward_hallucination(final_answer, doc_texts),   # penalty
        "entity":        reward_entity(final_answer, doc_texts),
        "entailment":    reward_entailment(final_answer, doc_texts),
        "fluency":       reward_fluency(final_answer),
        "conciseness":   reward_conciseness(final_answer),
        "query_align":   reward_query_alignment(question, final_answer),
    }


def combined_reward(rewards: Dict[str, float]) -> float:
    """
    Weighted combination of individual rewards.
    Hallucination is subtracted (it's a penalty signal).

    R = w_cor*cor + w_ret*ret - w_hal*hal + w_ent*ent
        + w_nli*entailment + w_flu*flu + w_con*con + w_alg*align
    """
    c = Config
    R = (
        c.RW_CORRECTNESS   * rewards.get("correctness",   0.0)
      + c.RW_RETRIEVAL     * rewards.get("retrieval",     0.0)
      - c.RW_HALLUCINATION * rewards.get("hallucination", 0.0)
      + c.RW_ENTITY        * rewards.get("entity",        0.0)
      + c.RW_ENTAILMENT    * rewards.get("entailment",    0.0)
      + c.RW_FLUENCY       * rewards.get("fluency",       0.0)
      + c.RW_CONCISENESS   * rewards.get("conciseness",   0.0)
      + c.RW_QUERY_ALIGN   * rewards.get("query_align",   0.0)
    )
    return float(np.clip(R, -1.0, 1.0))


# ── Utils ─────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
