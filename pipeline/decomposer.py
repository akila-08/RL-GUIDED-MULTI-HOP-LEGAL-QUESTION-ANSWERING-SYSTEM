"""
pipeline/decomposer.py
----------------------
T5-based question decomposer with quality evaluation and rule-based fallback.

Flow
────
1. decompose(question)
   └─ t5_decompose(question)          ← Flan-T5-base from decomp_model/
   └─ evaluate_decomposition(...)     ← ROUGE-L + keyword coverage + count + atomicity
   └─ if score poor → baseline_decompose(question)   ← rule-based fallback

Evaluation metrics
──────────────────
rouge_score : ROUGE-L recall of sub-qs joined, vs original question
              (high = sub-qs collectively paraphrase the question)
coverage    : fraction of key tokens from question found in sub-qs
              (ensures no important concept is dropped)
Threshold   : configurable in Config (DECOMP_ROUGE_THRESH, DECOMP_COVERAGE_THRESH)
"""

from __future__ import annotations

import os
import re
import logging
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from core.config import Config
from rl.actions import DecomposeResult

log = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_tokenizer:   Optional[AutoTokenizer]          = None
_t5_model:    Optional[AutoModelForSeq2SeqLM]  = None
_rouge_scorer: Optional[object]                = None

STOPWORDS = {
    "how", "does", "do", "is", "are", "the", "of", "and", "or", "to", "in",
    "for", "what", "when", "if", "a", "an", "their", "they", "together",
    "under", "with", "from", "by", "on", "at", "this", "that", "these",
    "those", "it", "as", "be", "which", "have", "has", "had", "been",
    "would", "could", "should",
}


def _get_t5():
    global _tokenizer, _t5_model
    if _tokenizer is None or _t5_model is None:
        model_path = Config.DECOMP_MODEL_PATH

        # Guard: if the folder doesn't exist, fail fast with a clear message
        # so decompose() can catch it and use the rule-based fallback.
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"T5 decomposer model not found at '{model_path}'. "
                "Train the decomposer first (scripts/train_decomposer.py) "
                "or place a fine-tuned Flan-T5 checkpoint there. "
                "Falling back to rule-based decomposer."
            )

        log.info("Loading T5 decomposer from: %s", model_path)
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Add special tokens that were added during training
        special_tokens = {"additional_special_tokens": ["<rule>", "<apply>"]}
        _tokenizer.add_special_tokens(special_tokens)

        _t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        _t5_model.resize_token_embeddings(len(_tokenizer))
        _t5_model.eval()
    return _tokenizer, _t5_model


def _rouge_instance():
    global _rouge_scorer
    if _rouge_scorer is None:
        try:
            from rouge_score import rouge_scorer as rs
            _rouge_scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            log.warning("rouge_score not installed; ROUGE eval disabled.")
    return _rouge_scorer


# ── T5 Inference ──────────────────────────────────────────────────────────────

def t5_decompose(question: str) -> List[str]:
    """
    Run the fine-tuned Flan-T5 model to decompose a complex legal question.

    Input format : "decompose: <question>"
    Output format: "<rule> sub-q1\\n<rule> sub-q2\\n<apply> synthesis-q"

    Returns a list of sub-question strings (cleaned, without tags).
    """
    tokenizer, model = _get_t5()
    device = next(model.parameters()).device

    input_text = f"decompose: {question.strip()}"
    inputs = tokenizer(
        input_text,
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return _parse_t5_output(raw)


def _parse_t5_output(raw: str) -> List[str]:
    """Parse tagged T5 output into clean sub-question strings."""
    # Remove padding/eos tokens
    raw = re.sub(r'</?pad>|<eos>|<s>|</s>', '', raw)
    # Split on newlines or <rule>/<apply> tags
    parts = re.split(r'<rule>|<apply>', raw)
    sub_qs = []
    for p in parts:
        p = p.strip().strip('"').strip()
        if len(p) > 10 and '?' in p:
            sub_qs.append(p)
    return sub_qs


# ── Decomposition Evaluation ──────────────────────────────────────────────────

def evaluate_decomposition(
    question: str,
    sub_questions: List[str],
) -> Tuple[float, float, bool]:
    """
    Evaluate quality of a decomposition.

    Returns
    -------
    rouge_score    : float in [0, 1]
    coverage_score : float in [0, 1]
    is_good        : bool — True if both thresholds are met AND count is valid
    """
    if not sub_questions:
        return 0.0, 0.0, False

    # 1. Count check
    n = len(sub_questions)
    count_ok = Config.DECOMP_MIN_SUBQ <= n <= Config.DECOMP_MAX_SUBQ

    # 2. ROUGE-L (sub_qs collectively paraphrase the question)
    rouge_score = 0.0
    scorer = _rouge_instance()
    if scorer is not None:
        joined = " ".join(sub_questions)
        try:
            result     = scorer.score(question, joined)
            rouge_score = float(result["rougeL"].recall)
        except Exception as e:
            log.warning("ROUGE eval error: %s", e)

    # 3. Keyword coverage
    q_tokens = {
        w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', question)
        if w.lower() not in STOPWORDS
    }
    sub_blob = " ".join(sub_questions).lower()
    if q_tokens:
        found    = sum(1 for t in q_tokens if t in sub_blob)
        coverage = found / len(q_tokens)
    else:
        coverage = 1.0   # no meaningful tokens to check

    # 4. Atomicity: non-final sub-qs should be ≤ 25 words and single question
    atomic_ok = True
    for sq in sub_questions[:-1]:
        if sq.count('?') > 1 or len(sq.split()) > 25:
            atomic_ok = False
            break

    is_good = (
        count_ok
        and rouge_score  >= Config.DECOMP_ROUGE_THRESH
        and coverage     >= Config.DECOMP_COVERAGE_THRESH
        and atomic_ok
    )

    log.debug(
        "DecompEval: rouge=%.3f coverage=%.3f count_ok=%s atomic=%s → good=%s",
        rouge_score, coverage, count_ok, atomic_ok, is_good,
    )
    return rouge_score, coverage, is_good


# ── Baseline (Rule-based) Fallback ────────────────────────────────────────────

def baseline_decompose(question: str) -> List[str]:
    """
    Rule-based decomposer extracted from scripts/baseline_decomposer.py.
    Used as fallback when T5 quality is insufficient.
    """
    try:
        # Import the clean, standalone version of the heuristic rules
        from pipeline.baseline_rules import dataset_style_decompose_v3
        result = dataset_style_decompose_v3(question)
        return result.get("sub_questions", [])
    except Exception as e:
        log.warning("Baseline decomposer unavailable (%s) — using ultimate fallback.", e)
        # Ultimate fallback: split question into 2 simple sub-questions
        return [
            f"What is the legal rule relevant to: {question}",
            f"How does that rule apply to answer: {question}",
        ]


# ── Public API ────────────────────────────────────────────────────────────────

def decompose(question: str) -> DecomposeResult:
    """
    Main entry point. Attempts T5 decomposition; falls back to baseline
    if quality check fails OR if T5 model is not available.

    Returns a DecomposeResult with sub_questions and evaluation metadata.
    """
    # --- Step A: T5 attempt (skipped if model folder does not exist) ---
    try:
        sub_qs_t5 = t5_decompose(question)
        rouge, coverage, is_good = evaluate_decomposition(question, sub_qs_t5)

        if is_good:
            log.info(
                "T5 decomposition accepted: %d sub-qs (rouge=%.3f coverage=%.3f)",
                len(sub_qs_t5), rouge, coverage,
            )
            return DecomposeResult(
                sub_questions  = sub_qs_t5,
                used_baseline  = False,
                rouge_score    = rouge,
                coverage_score = coverage,
            )

        log.info(
            "T5 quality below threshold (rouge=%.3f coverage=%.3f); using baseline.",
            rouge, coverage,
        )

    except FileNotFoundError as e:
        log.warning("T5 model unavailable — using rule-based fallback. (%s)", e)

    except Exception as e:
        log.error("T5 decomposition failed unexpectedly: %s — using baseline.", e)

    # --- Step B: Baseline fallback (rule-based) ---
    sub_qs_base = baseline_decompose(question)
    rouge_b, cov_b, _ = evaluate_decomposition(question, sub_qs_base)

    return DecomposeResult(
        sub_questions  = sub_qs_base,
        used_baseline  = True,
        rouge_score    = rouge_b,
        coverage_score = cov_b,
    )
