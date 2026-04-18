from __future__ import annotations

import re
import logging
from typing import Optional, Callable

log = logging.getLogger(__name__)

# ── Question-type complexity floors ───────────────────────────────────────────
# These are MINIMUM scores per question type.
# The actual score = max(classifier_raw, floor + lexical_bonus)
_TYPE_FLOOR: dict[str, float] = {
    "comparative": 0.80,   # always needs 2 concepts + comparison
    "conditional": 0.65,   # needs condition + consequence hops
    "analytical":  0.50,   # single-hop factual
}

# ── Lexical complexity signals ────────────────────────────────────────────────

# Words that signal multi-hop reasoning requirements
_MULTI_HOP_SIGNALS = [
    "and why", "and how", "explain", "discuss", "in what way",
    "what are the implications", "both", "neither", "as well as",
    "furthermore", "moreover", "in addition", "in relation to",
    "what happens when", "under what circumstances",
]

# Clause-boundary markers (more clauses → harder)
_CLAUSE_MARKERS = [",", ";", "and", "or", "but", "while", "whereas",
                   "although", "however", "unless", "if", "when", "because"]


def _lexical_bonus(question: str) -> float:
    q = question.lower()
    bonus = 0.0

    # Multi-hop signal count (capped at 0.10)
    hop_hits = sum(1 for sig in _MULTI_HOP_SIGNALS if sig in q)
    bonus += min(hop_hits * 0.05, 0.10)

    # Clause marker count beyond the first 2
    clause_hits = sum(1 for m in _CLAUSE_MARKERS if m in q.split())
    bonus += min(max(clause_hits - 2, 0) * 0.01, 0.05)

    # Length signal
    if len(question.split()) > 15:
        bonus += 0.05

    return round(min(bonus, 0.20), 4)


def compute_complexity_score(
    question: str,
    classifier_fn: Optional[Callable[[str], float]] = None,
) -> float:
    from pipeline.baseline_rules import infer_question_type

    # Step 1: Raw classifier score
    if classifier_fn is not None:
        try:
            raw = float(classifier_fn(question))
        except Exception as e:
            log.warning("Classifier error: %s — using 0.7", e)
            raw = 0.7
    else:
        raw = 0.7

    # Step 2: Question type + floor
    q_type = infer_question_type(question)
    floor  = _TYPE_FLOOR.get(q_type, 0.50)

    # Step 3: Lexical bonus
    bonus  = _lexical_bonus(question)

    # Step 4: Final score
    score  = max(raw, floor + bonus)
    score  = round(min(score, 1.0), 4)

    log.debug(
        "complexity | q_type=%-12s | raw=%.3f | floor=%.2f | bonus=%.3f | final=%.3f",
        q_type, raw, floor, bonus, score,
    )
    return score


def explain_complexity(question: str, classifier_fn=None) -> dict:
    from pipeline.baseline_rules import infer_question_type

    q_type = infer_question_type(question)
    floor  = _TYPE_FLOOR.get(q_type, 0.50)
    bonus  = _lexical_bonus(question)

    if classifier_fn is not None:
        try:
            raw = float(classifier_fn(question))
        except Exception:
            raw = 0.7
    else:
        raw = 0.7

    final = round(min(max(raw, floor + bonus), 1.0), 4)
    verdict = "simple" if final < 0.55 else ("moderate" if final < 0.75 else "complex")

    return {
        "question_type":  q_type,
        "classifier_raw": round(raw, 4),
        "type_floor":     floor,
        "lexical_bonus":  bonus,
        "final_score":    final,
        "verdict":        verdict,
    }
