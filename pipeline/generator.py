"""
pipeline/generator.py
---------------------
Sub-answer generation using Google Gemini (primary) with Anthropic Claude
as fallback. Controlled by LLM_MODEL in config / .env.

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


# ── LLM backends ─────────────────────────────────────────────────────────────

import time

def _call_gemini(prompt: str, temperature: float, retries: int = 3) -> str:
    """Call Google Gemini API using the new google.genai SDK with automatic retry for rate limits."""
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError

        client = genai.Client(api_key=Config.GEMINI_API_KEY)
        
        for attempt in range(retries):
            try:
                # Add a small base delay to help respect the 15 Requests Per Minute free tier limit
                time.sleep(4) 
                
                response = client.models.generate_content(
                    model=Config.LLM_MODEL,
                    contents=f"{_SYSTEM_PROMPT}\n\n{prompt}",
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=Config.GEN_MAX_TOKENS,
                    ),
                )
                return response.text.strip()
            except APIError as e:
                # 429 means Resource Exhausted / Quota Limit
                if e.code == 429 and attempt < retries - 1:
                    wait_time = 25  # Wait 25 seconds before retrying
                    log.warning(f"Rate limit hit! Sleeping for {wait_time}s before attempt {attempt+2}...")
                    time.sleep(wait_time)
                else:
                    raise e

    except Exception as e:
        log.error("Gemini call failed: %s", e)
        return ""


def _call_openai(prompt: str, temperature: float) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
        import os
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set.")
            
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=Config.GEN_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("OpenAI call failed: %s", e)
        return ""


def _call_groq(prompt: str, temperature: float) -> str:
    """Call Groq API (Fast Llama 3)."""
    try:
        from groq import Groq
        import os
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not set.")
            
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Groq specific active models
        model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        if "gemini" in model_name or "claude" in model_name or "gpt" in model_name or "llama3-8b-8192" in model_name:
            model_name = "llama-3.1-8b-instant"  # Automatically fix model name for Groq

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=Config.GEN_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error("Groq call failed: %s", e)
        return ""


def _call_llm(prompt: str, temperature: float) -> str:
    """
    Route to the correct LLM backend based on config.

    Priority:
      1. If GROQ_API_KEY is set    → use Groq
      2. If OPENAI_API_KEY is set  → use OpenAI
      3. If GEMINI_API_KEY is set  → use Gemini
      4. If ANTHROPIC_API_KEY set  → use Claude
      5. None                      → Dummy Fallback
    """
    import os
    if os.getenv("GROQ_API_KEY"):
        result = _call_groq(prompt, temperature)
        if result: return result

    if Config.GEMINI_API_KEY:
        result = _call_gemini(prompt, temperature)
        if result: return result
        # Gemini failed — try Claude if available
        if Config.ANTHROPIC_API_KEY:
            log.warning("Gemini failed, falling back to Anthropic Claude.")
            return _call_anthropic(prompt, temperature)
        return ""

    if Config.ANTHROPIC_API_KEY:
        return _call_anthropic(prompt, temperature)

    log.error(
        "No LLM API key configured. "
        "Set OPENAI_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY in your .env file."
    )
    return ""


# ── Core generation ───────────────────────────────────────────────────────────

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
        meta  = c.get("metadata", {})
        art   = meta.get("article_num", "?")
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
    context  = _build_context(chunks)
    prompt   = _USER_TEMPLATE.format(context=context, sub_question=sub_question)
    temp     = Config.GEN_TEMPERATURE_DEFAULT
    answer   = _call_llm(prompt, temp)
    retried  = False

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
        n_chunks = len(chunks)
        if n_chunks == 0:
            assigned_chunks: List[Dict] = []
        elif n_chunks <= Config.TOP_K_RERANK:
            assigned_chunks = chunks
        else:
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
