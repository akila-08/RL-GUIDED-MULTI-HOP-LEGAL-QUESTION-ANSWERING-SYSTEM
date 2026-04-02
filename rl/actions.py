"""
rl/actions.py
-------------
Macro-action definitions and sub-action dispatch logic.

The RL agent outputs ONE of 4 macro-action IDs (0-3).
Within each macro, sub-actions are triggered automatically based on
intermediate signals (decomp quality, retrieval confidence, etc.)
keeping the agent's action space small and fast to learn.

Macro-actions
─────────────
0  DECOMPOSE  → sub: decompose | evaluate | redecompose
1  RETRIEVE   → sub: fetch_top_k | rerank | reformulate_query
2  GENERATE   → sub: set_temperature | retry_revised_prompt
3  COMBINE    → sub: concatenate | summarise
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Dict, Optional, Tuple, cast, Iterable

import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Macro-action enum
# ---------------------------------------------------------------------------

class MacroAction(IntEnum):
    DECOMPOSE = 0
    RETRIEVE  = 1
    GENERATE  = 2
    COMBINE   = 3

    @classmethod
    def name_of(cls, idx: int) -> str:
        return str(cls(idx).name)


# ---------------------------------------------------------------------------
# Sub-action result containers
# ---------------------------------------------------------------------------

class DecomposeResult:
    """Outcome of the DECOMPOSE macro-action."""
    def __init__(
        self,
        sub_questions: List[str],
        used_baseline: bool,
        rouge_score: float,
        coverage_score: float,
    ):
        self.sub_questions   = sub_questions
        self.used_baseline   = used_baseline
        self.rouge_score     = rouge_score
        self.coverage_score  = coverage_score

    def __repr__(self):
        return (
            f"DecomposeResult(n={len(self.sub_questions)}, "
            f"rouge={self.rouge_score:.3f}, coverage={self.coverage_score:.3f}, "
            f"fallback={self.used_baseline})"
        )


class RetrieveResult:
    """Outcome of the RETRIEVE macro-action."""
    def __init__(
        self,
        chunks: List[Dict],
        was_reformulated: bool,
        reformulated_queries: Optional[List[str]] = None,
    ):
        self.chunks               = chunks
        self.was_reformulated     = was_reformulated
        self.reformulated_queries = reformulated_queries or []

    @property
    def texts(self) -> List[str]:
        return [c.get("text", "") for c in self.chunks]

    def __repr__(self):
        return (
            f"RetrieveResult(n_chunks={len(self.chunks)}, "
            f"reformulated={self.was_reformulated})"
        )


class GenerateResult:
    """Outcome of the GENERATE macro-action."""
    def __init__(
        self,
        sub_answers: List[str],
        temperatures_used: List[float],
        retried: bool,
    ):
        self.sub_answers        = sub_answers
        self.temperatures_used  = temperatures_used
        self.retried            = retried

    def __repr__(self):
        return (
            f"GenerateResult(n={len(self.sub_answers)}, "
            f"retried={self.retried})"
        )


class CombineResult:
    """Outcome of the COMBINE macro-action."""
    def __init__(
        self,
        final_answer: str,
        method: str,   # 'concatenate' or 'summarise'
    ):
        self.final_answer = final_answer
        self.method       = method

    def __repr__(self):
        return f"CombineResult(method={self.method}, len={len(self.final_answer)})"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch(
    macro_action: int,
    *,
    question: str,
    sub_questions: List[str],
    retrieve_result: Optional[RetrieveResult],
    generate_result: Optional[GenerateResult],
    complexity_score: float,
    step: int,
    # injected pipeline modules (passed from env to avoid circular imports)
    decomposer,
    retriever,
    generator,
    combiner,
) -> Tuple:
    """
    Execute a macro-action and return (result, log_msg).

    Parameters
    ----------
    macro_action      : integer in {0,1,2,3}
    question          : original question string
    sub_questions     : current decomposed sub-questions
    retrieve_result   : previous retrieve result (or None)
    generate_result   : previous generate result (or None)
    complexity_score  : classifier confidence
    step              : current episode step
    decomposer        : pipeline.decomposer module
    retriever         : pipeline.retriever module
    generator         : pipeline.generator module
    combiner          : pipeline.combiner module

    Returns
    -------
    (result_object, str_log_message)
    """
    action = MacroAction(macro_action)

    if action == MacroAction.DECOMPOSE:
        result = decomposer.decompose(question)
        log.info("[DECOMPOSE] %s", result)
        return result, f"Decomposed into {len(result.sub_questions)} sub-questions"

    elif action == MacroAction.RETRIEVE:
        queries = sub_questions if sub_questions else [question]
        # Decide whether to reformulate based on step & complexity
        # Reformulate if: step > 1 AND no good docs yet OR high complexity + step 0
        should_reformulate = (
            step > 1 and (retrieve_result is None or len(retrieve_result.chunks) == 0)
        ) or (
            complexity_score > 0.8 and step == 0
        )
        result = retriever.retrieve(queries, reformulate=should_reformulate)
        log.info("[RETRIEVE] %s", result)
        return result, f"Retrieved {len(result.chunks)} chunks"

    elif action == MacroAction.GENERATE:
        if not sub_questions:
            sub_questions = [question]
        docs = retrieve_result.chunks if retrieve_result else []
        result = generator.generate(sub_questions, docs)
        log.info("[GENERATE] %s", result)
        return result, f"Generated {len(result.sub_answers)} sub-answers"

    elif action == MacroAction.COMBINE:
        answers = generate_result.sub_answers if generate_result else []
        # Choose summarise for complex questions (high c_t), concatenate for simple
        method = "summarise" if complexity_score >= 0.5 else "concatenate"
        result = combiner.combine(answers, question, method=method)
        log.info("[COMBINE] %s", result)
        return result, f"Combined via {result.method}"

    else:
        raise ValueError(f"Unknown macro_action: {macro_action}")


ACTION_NAMES = [
    action.name for action in cast(Iterable[MacroAction], MacroAction)
]