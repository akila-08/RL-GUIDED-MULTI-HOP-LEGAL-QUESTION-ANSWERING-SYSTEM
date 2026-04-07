"""
pipeline/combiner.py
--------------------
Combines sub-answers into a single final answer.

Two sub-actions (chosen by the RL environment based on complexity_score):
  concatenate : simple join — preserves all detail, fast, no LLM call
  summarise   : LLM-powered condensation — better for complex multi-hop answers

The RL environment chooses:
  summarise   → complexity_score >= 0.5  (complex question)
  concatenate → complexity_score <  0.5  (simple question)
"""

from __future__ import annotations

import logging
from typing import List

from core.config import Config
from rl.actions import CombineResult

log = logging.getLogger(__name__)

_SUMMARISE_SYSTEM = (
    "You are a legal assistant specialised in the Constitution of India. "
    "Synthesise the provided sub-answers into one clear, concise, and coherent "
    "final answer to the original question. Use legal language, cite article "
    "numbers where relevant, and avoid repetition."
)

_SUMMARISE_USER = """\
### Original Question:
{question}

### Sub-Answers:
{sub_answers_text}

### Final Answer (synthesised):"""


def _concatenate(sub_answers: List[str], question: str) -> str:
    """
    Simple concatenation with ordinal prefixes.
    Filters out empty / 'not found' answers.
    """
    parts = []
    for i, ans in enumerate(sub_answers, 1):
        if ans and "not found" not in ans.lower():
            parts.append(f"[{i}] {ans.strip()}")
    if not parts:
        return "Unable to retrieve relevant information for this question."
    return "\n\n".join(parts)
def _summarise(sub_answers: List[str], question: str) -> str:
    """
    Uses LLM (Groq) to synthesise sub-answers into one coherent final answer.
    Falls back to concatenation if LLM call fails.
    """
    import os

    # Filter vacuous answers
    valid = [a.strip() for a in sub_answers if a and "not found" not in a.lower()]
    if not valid:
        return "Unable to retrieve relevant information for this question."

    sub_answers_text = "\n\n".join(
        f"{i}. {a}" for i, a in enumerate(valid, 1)
    )
    prompt = _SUMMARISE_USER.format(
        question=question,
        sub_answers_text=sub_answers_text,
    )

    try:
        from groq import Groq
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not set in .env")
            
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Format model name correctly
        model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        if "gemini" in model_name or "claude" in model_name or "gpt" in model_name or "llama3-8b-8192" in model_name:
            model_name = "llama-3.1-8b-instant"

        response = client.chat.completions.create(
            model=model_name,
            max_tokens=Config.GEN_MAX_TOKENS,
            temperature=Config.GEN_TEMPERATURE_DEFAULT,
            messages=[
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.warning("Summarisation LLM call failed (%s); falling back to concat.", e)
        return _concatenate(sub_answers, question)


# ── Public API ────────────────────────────────────────────────────────────────

def combine(
    sub_answers: List[str],
    question: str,
    method: str = "summarise",
) -> CombineResult:
    """
    Combine sub-answers into a final answer.

    Parameters
    ----------
    sub_answers : list of sub-answer strings from the generator
    question    : original user question (used as context for summarisation)
    method      : 'concatenate' or 'summarise'

    Returns
    -------
    CombineResult with .final_answer and .method
    """
    if not sub_answers:
        return CombineResult(
            final_answer="No sub-answers were generated.",
            method="concatenate",
        )

    if method == "summarise":
        answer = _summarise(sub_answers, question)
    else:
        answer = _concatenate(sub_answers, question)

    log.info("Combiner [%s]: final answer length=%d chars", method, len(answer))
    return CombineResult(final_answer=answer, method=method)
