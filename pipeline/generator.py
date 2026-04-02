"""
pipeline/generator.py
---------------------
Sub-answer generation using Claude (Anthropic) with temperature control.

Sub-actions supported (selected automatically based on context):
  set_temperature      : use DEFAULT (0.3) for factual grounding
  retry_revised_prompt : switch to RETRY (0.7) + richer prompt if first
                         generation is empty / too short / "Not found"

Each sub-question is answered independently using its relevant chunks.
If fewer chunks than sub-questions exist, the available chunks are shared.
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Optional

import anthropic

from core.config import Config
from rl.actions import GenerateResult

log = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a legal assistant specialised in the Constitution of India. "
    "Answer questions strictly based on the provided context. "
    "If the answer is not in the context, say 'Not found in the provided context.' "
    "Be precise, cite article numbers when relevant, and stay concise."
)

_USER_TEMPLATE = """\
### Context (retrieved constitutional provisions):
{context}

### Sub-Question:
{sub_question}

### Answer:"""

_RETRY_TEMPLATE = """\
### Context (retrieved constitutional provisions):
{context}

### Sub-Question:
{sub_question}

The previous answer was insufficient. Please provide a more detailed and complete \
answer, elaborating on the constitutional provisions, their scope, and their \
implications. If the context lacks information, explicitly say so.

### Improved Answer:"""


# ── Anthropic client singleton ────────────────────────────────────────────────

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = Config.ANTHROPIC_API_KEY
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ── Core generation ───────────────────────────────────────────────────────────

def _call_llm(prompt: str, temperature: float) -> str:
    """Single LLM call; returns stripped text or empty string on error."""
    try:
        client  = _get_client()
        message = client.messages.create(
            model       = Config.LLM_MODEL,
            max_tokens  = Config.GEN_MAX_TOKENS,
            temperature = temperature,
            system      = _SYSTEM_PROMPT,
            messages    = [{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        log.error("LLM call failed: %s", e)
        return ""


def _is_bad_answer(answer: str) -> bool:
    """Heuristic: is the answer insufficient?"""
    if not answer or len(answer.split()) < 8:
        return True
    BAD_PHRASES = [
        "not found in the provided context",
        "i cannot",
        "i don't know",
        "no information",
        "not available",
    ]
    return any(p in answer.lower() for p in BAD_PHRASES)


def _build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks as context string for the prompt."""
    if not chunks:
        return "No context retrieved."
    parts = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        art  = meta.get("article_num", "?")
        title = meta.get("title", "")[:60]
        text  = c.get("text", "")[:600]
        parts.append(f"[{i}] Article {art} – {title}\n{text}")
    return "\n\n".join(parts)


def _generate_one(
    sub_question: str,
    chunks: List[Dict],
) -> tuple[str, float, bool]:
    """
    Generate answer for a single sub-question.

    Returns (answer_text, temperature_used, was_retried).
    """
    context   = _build_context(chunks)
    prompt    = _USER_TEMPLATE.format(context=context, sub_question=sub_question)
    temp      = Config.GEN_TEMPERATURE_DEFAULT
    answer    = _call_llm(prompt, temp)
    retried   = False

    if _is_bad_answer(answer):
        log.info("First attempt insufficient — retrying with revised prompt.")
        retry_prompt = _RETRY_TEMPLATE.format(context=context, sub_question=sub_question)
        temp         = Config.GEN_TEMPERATURE_RETRY
        answer       = _call_llm(retry_prompt, temp)
        retried      = True

    return answer, temp, retried


# ── Public API ────────────────────────────────────────────────────────────────

def generate(
    sub_questions: List[str],
    chunks: List[Dict],
) -> GenerateResult:
    """
    Generate sub-answers for each sub-question using retrieved chunks.

    Strategy
    ────────
    - Assign chunks to sub-questions round-robin so every question gets context
    - Attempt generation at temperature DEFAULT (0.3)
    - If the answer is bad  → retry at RETRY temperature (0.7) with a richer prompt

    Returns a GenerateResult with sub_answers, temperatures_used, retried flag.
    """
    if not sub_questions:
        return GenerateResult(sub_answers=[], temperatures_used=[], retried=False)

    sub_answers:       List[str]   = []
    temperatures_used: List[float] = []
    any_retried = False

    for i, sq in enumerate(sub_questions):
        # Distribute chunks: each sub-question gets a window of chunks
        n_chunks = len(chunks)
        if n_chunks == 0:
            assigned_chunks: List[Dict] = []
        elif n_chunks <= Config.TOP_K_RERANK:
            # All sub-questions share all chunks
            assigned_chunks = chunks
        else:
            # Sliding window: each sub-question gets a proportional subset
            window = max(2, n_chunks // len(sub_questions))
            start  = (i * window) % n_chunks
            assigned_chunks = chunks[start: start + window]

        answer, temp, retried = _generate_one(sq, assigned_chunks)

        sub_answers.append(answer)
        temperatures_used.append(temp)
        if retried:
            any_retried = True

        log.debug("Sub-Q[%d] answer (temp=%.1f, retry=%s): %s…",
                  i, temp, retried, answer[:80])

    return GenerateResult(
        sub_answers       = sub_answers,
        temperatures_used = temperatures_used,
        retried           = any_retried,
    )
