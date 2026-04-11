"""
rl/rewards.py
-------------
8 reward signal functions for the RL-Guided Legal QA system.

Signal          Direction  Description
────────────────────────────────────────────────────────────────────
retrieval         +pos      cosine sim between sub-q and chunks
groundedness      +pos      1 - unsupported entity fraction (inverted hallucination)
entity            +pos      article/entity coverage in retrieved docs
entailment        +pos      NLI entailment score (context → answer)
fluency           +pos      grammar / language quality score
conciseness       +pos      length efficiency (avg of sub-answer + final answer)
query_align       +pos      semantic similarity answer ↔ original question
decomposition     +pos      appropriateness of question decomposition given complexity

All individual rewards are in [0, 1]. Combined reward:
  R = w_gnd*gnd + w_nli*nli + w_alg*align + w_ret*ret
      + w_ent*ent + w_dec*dec + w_flu*flu + w_con*con

NOTE on groundedness:
  groundedness = 1 - hallucination_fraction
  So a fully grounded answer (no hallucination) → 1.0 reward.
  A fully hallucinated answer → 0.0 reward.
  All signals are now purely additive — no subtraction needed.

NOTE on conciseness:
  Scored separately for sub-answers (vs Config.MAX_SUB_ANSWER_LEN)
  and the final answer (vs Config.MAX_FINAL_ANSWER_LEN), then averaged.
  Both granular scores are also exposed in the rewards dict for logging.
    Recommended values:
      MAX_SUB_ANSWER_LEN   = 350   chars  (~2-3 focused sentences)
      MAX_FINAL_ANSWER_LEN = 1000  chars  (~synthesis + conditions)
  Set these to your p90 answer length from real data for best calibration.

NOTE on decomposition:
  Rewards whether decomposition depth MATCHES question complexity.
  Complexity is read directly from the RL state's classification model
  confidence scores ({"simple": p, "complex": 1-p}) — no heuristics.
  Simple question + no decomposition    → high reward (correct behaviour)
  Simple question + over-decomposition  → low reward  (unnecessary splits)
  Complex question + under-decomposition → low reward (missed structure)
  Complex question + good decomposition  → high reward
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _complexity_from_classifier(complexity_scores: Dict[str, float]) -> float:
    """
    Extract complexity in [0, 1] from the RL state's classifier output.

    Expects complexity_scores to be a dict like:
      {"simple": 0.85, "complex": 0.15}   -> simple question  -> 0.15
      {"simple": 0.10, "complex": 0.90}   -> complex question -> 0.90

    The 'complex' confidence IS the complexity score directly —
    no transformation needed since it's already a calibrated probability.

    Falls back to 0.5 (moderate/neutral) if scores are missing or malformed.
    """
    if not complexity_scores:
        log.warning("complexity_scores missing from state; defaulting to 0.5")
        return 0.5
    try:
        complex_conf = float(complexity_scores.get("c_t", 0.5))
        return float(np.clip(complex_conf, 0.0, 1.0))
    except (TypeError, ValueError) as e:
        log.warning("Invalid complexity_scores format: %s — defaulting to 0.5", e)
        return 0.5


def _expected_sub_question_range(complexity: float):
    """
    Map complexity score to (min_expected, max_expected) sub-question count.

    complexity < 0.25  -> simple:   [1, 1]
    complexity < 0.55  -> moderate: [2, 3]
    complexity >= 0.55 -> complex:  [3, 6]
    """
    if complexity < 0.25:
        return (1, 1)
    elif complexity < 0.55:
        return (2, 3)
    else:
        return (3, 6)


# ── Individual reward functions ───────────────────────────────────────────────

def reward_retrieval(sub_questions: List[str], doc_texts: List[str]) -> float:
    """
    Mean cosine similarity between sub-question embeddings and
    the mean chunk embedding. Measures how relevant retrieved docs are.
    """
    if not sub_questions or not doc_texts:
        return 0.0
    q_vecs   = _embed_texts(sub_questions)   # (n_q, 384)
    doc_vecs = _embed_texts(doc_texts)       # (n_d, 384)
    q_mean   = q_vecs.mean(axis=0)
    d_mean   = doc_vecs.mean(axis=0)
    sim      = _cosine(q_mean, d_mean)
    # Map cosine sim from [-1, 1] to [0, 1]
    return float(np.clip((sim + 1) / 2, 0.0, 1.0))


def reward_groundedness(final_answer: str, doc_texts: List[str]) -> float:
    """
    Inverted hallucination signal: fraction of answer entities that ARE
    supported by retrieved documents.

    groundedness = 1 - hallucination_fraction

    Returns value in [0, 1]:
      1.0 -> all entities grounded in context (no hallucination)
      0.0 -> all entities unsupported (fully hallucinated)

    This is a purely positive reward — higher is always better.
    No subtraction needed in combined_reward.

    Neutral fallback (0.5) is used when no entities can be extracted,
    since absence of entities is neither good nor bad evidence.
    """
    if not final_answer or not doc_texts:
        return 0.5  # conservative neutral: no evidence either way
    context_blob = " ".join(doc_texts).lower()
    entities     = _extract_entities(final_answer)
    if not entities:
        return 0.5  # no extractable entities -> neutral
    unsupported            = sum(1 for e in entities if e not in context_blob)
    hallucination_fraction = unsupported / len(entities)
    return float(1.0 - hallucination_fraction)


def reward_entity(final_answer: str, doc_texts: List[str]) -> float:
    """
    Fraction of entities/article refs in the answer that appear in
    the retrieved context. Rewards entity consistency.
    """
    if not final_answer or not doc_texts:
        return 0.0
    context_blob = " ".join(doc_texts).lower()
    entities     = _extract_entities(final_answer)
    if not entities:
        return 0.5   # no entities -> neutral
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

    score   = 1.0
    words   = final_answer.split()
    n_words = len(words)

    # Too short
    if n_words < 10:
        score -= 0.3

    # Does not end with punctuation
    if final_answer.strip()[-1] not in ".!?":
        score -= 0.15

    # Repetition: fraction of duplicate bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
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


def reward_conciseness(answer: str, max_len: int) -> float:
    """
    Length efficiency: penalise answers that exceed their expected length ceiling.
    Score = 1 - (len / max_len), clipped to [0, 1].

    Args:
      answer:   The answer text to evaluate.
      max_len:  Length ceiling appropriate for this answer type:
                  Config.MAX_SUB_ANSWER_LEN   for sub-answers  (recommended: 350 chars)
                  Config.MAX_FINAL_ANSWER_LEN for final answer (recommended: 1000 chars)

    Called twice per episode in compute_all_rewards — once for sub-answers
    (averaged across all of them) and once for the final answer. The two
    scores are then averaged into a single "conciseness" signal.
    """
    if not answer:
        return 0.0
    ratio = len(answer) / max(max_len, 1)
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def reward_query_alignment(question: str, final_answer: str) -> float:
    """
    Cosine similarity between the original question embedding and the
    final answer embedding. Ensures the answer is on-topic.
    """
    if not question or not final_answer:
        return 0.0
    q_vec = _embed_text(question)
    a_vec = _embed_text(final_answer)
    sim   = _cosine(q_vec, a_vec)
    return float(np.clip((sim + 1) / 2, 0.0, 1.0))


def reward_decomposition(
    sub_questions: List[str],
    complexity_scores: Dict[str, float],
) -> float:
    """
    Rewards APPROPRIATE decomposition relative to question complexity,
    using the RL state's classification model confidence scores directly.

    Args:
      sub_questions:      List of sub-questions produced by the decomposer.
      complexity_scores:  Dict from the RL state classifier, e.g.
                            {"simple": 0.85, "complex": 0.15}
                          The 'complex' confidence is used as the complexity
                          estimate in [0, 1] — no heuristics needed.

    The key insight: not decomposing a simple question is CORRECT behaviour
    and should be rewarded just as much as decomposing a complex one.

    Steps:
      1. Read complexity from classifier: complex_confidence in [0, 1].
      2. Map to expected sub-question count range.
      3. Score based on how well actual count falls within that range.

    Scoring:
      - Actual count within [min_expected, max_expected]  -> 1.0
      - Each step outside the range applies a 0.2 penalty.
      - Minimum score is 0.0.

    Examples (using classifier confidence):
      {"simple": 0.92, "complex": 0.08}  -> complexity=0.08 -> expect [1, 1]
        sub_questions = ["What is Article 5?"]           -> 1.0  correct, kept simple
        sub_questions = ["...", "...", "...", "..."]      -> 0.4  over-decomposed

      {"simple": 0.05, "complex": 0.95}  -> complexity=0.95 -> expect [3, 6]
        sub_questions = ["...", "...", "..."]             -> 1.0  correctly decomposed
        sub_questions = ["..."]                           -> 0.6  under-decomposed

      {"simple": 0.45, "complex": 0.55}  -> complexity=0.55 -> expect [3, 6]
        sub_questions = ["...", "..."]                    -> 0.8  borderline
    """
    # Treat 0 sub-questions as 1 (original question is itself a sub-question)
    n_actual   = max(len(sub_questions), 1)
    complexity = _complexity_from_classifier(complexity_scores)
    lo, hi     = _expected_sub_question_range(complexity)

    if lo <= n_actual <= hi:
        return 1.0

    steps_off = (lo - n_actual) if n_actual < lo else (n_actual - hi)
    penalty   = steps_off * 0.20
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


# ── Combined reward ───────────────────────────────────────────────────────────

def compute_all_rewards(
    question: str,
    final_answer: str,
    sub_questions: List[str],
    doc_texts: List[str],
    complexity_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute all 8 rewards and return as a dict.
    Does NOT apply weights — weights are applied in combined_reward().

    All signals are in [0, 1] and purely positive. No gold answer required.

    Args:
      question:           Original user question.
      final_answer:       Final generated answer.
      sub_questions:      Sub-questions produced by the decomposer.
                          Also used as proxy sub-answer texts for conciseness
                          (replace with actual sub-answer strings if available
                          in your state — they will score more accurately).
      doc_texts:          Retrieved document chunks.
      complexity_scores:  Dict from the RL state's classification model, e.g.
                            {"simple": 0.85, "complex": 0.15}
                          If None, decomposition reward defaults to neutral 0.5.

    Conciseness breakdown (all three keys returned):
      "conciseness_sub"   — mean conciseness across all sub-answers scored
                            against Config.MAX_SUB_ANSWER_LEN (recommended: 350)
      "conciseness_final" — conciseness of the final answer scored against
                            Config.MAX_FINAL_ANSWER_LEN (recommended: 1000)
      "conciseness"       — simple average of the two; used in combined_reward

    Note: if sub_questions is empty (simple question, no decomposition),
    sub_conciseness defaults to 1.0 — not decomposing is correct behaviour
    and should not be penalised at the conciseness level either.
    """
    # ── Conciseness: scored separately at each level, then averaged ───────────
    sub_conciseness_scores = [
        reward_conciseness(sq, Config.MAX_SUB_ANSWER_LEN)
        for sq in sub_questions
    ] if sub_questions else []

    # No sub-questions (simple path) -> sub-level conciseness is perfect (1.0)
    sub_conciseness   = float(np.mean(sub_conciseness_scores)) if sub_conciseness_scores else 1.0
    final_conciseness = reward_conciseness(final_answer, Config.MAX_FINAL_ANSWER_LEN)
    conciseness       = float(np.mean([sub_conciseness, final_conciseness]))

    return {
        "retrieval":          reward_retrieval(sub_questions, doc_texts),
        "groundedness":       reward_groundedness(final_answer, doc_texts),
        "entity":             reward_entity(final_answer, doc_texts),
        "entailment":         reward_entailment(final_answer, doc_texts),
        "fluency":            reward_fluency(final_answer),
        "conciseness":        conciseness,
        "conciseness_sub":    sub_conciseness,    # diagnostic — for logging only
        "conciseness_final":  final_conciseness,  # diagnostic — for logging only
        "query_align":        reward_query_alignment(question, final_answer),
        "decomposition":      reward_decomposition(sub_questions, complexity_scores or {}),
    }


def combined_reward(rewards: Dict[str, float]) -> float:
    """
    Weighted combination of all 8 reward signals.
    All signals are additive — no subtraction needed since groundedness
    is already the inverted form of hallucination (higher = better).

    Only "conciseness" (the averaged value) is used here.
    "conciseness_sub" and "conciseness_final" are diagnostic keys only.

    Weight allocation (sum = 1.0):
      groundedness  0.25  — most critical: answer must be grounded in context
      entailment    0.20  — strongest faithfulness signal (NLI-based)
      query_align   0.20  — answer must address the actual question
      retrieval     0.15  — pipeline health: are we fetching the right docs?
      entity        0.10  — entity consistency with retrieved context
      decomposition 0.05  — appropriate question breakdown
      fluency       0.03  — language quality
      conciseness   0.02  — length efficiency (minor concern)

    R = w_gnd*gnd + w_nli*nli + w_alg*align + w_ret*ret
        + w_ent*ent + w_dec*dec + w_flu*flu + w_con*con
    """
    c = Config
    R = (
        c.RW_GROUNDEDNESS  * rewards.get("groundedness",  0.0)
      + c.RW_ENTAILMENT    * rewards.get("entailment",    0.0)
      + c.RW_QUERY_ALIGN   * rewards.get("query_align",   0.0)
      + c.RW_RETRIEVAL     * rewards.get("retrieval",     0.0)
      + c.RW_ENTITY        * rewards.get("entity",        0.0)
      + c.RW_DECOMPOSITION * rewards.get("decomposition", 0.0)
      + c.RW_FLUENCY       * rewards.get("fluency",       0.0)
      + c.RW_CONCISENESS   * rewards.get("conciseness",   0.0)
    )
    return float(np.clip(R, 0.0, 1.0))


# ── Utils ─────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()