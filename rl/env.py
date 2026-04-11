"""
rl/env.py
---------
LegalQAEnv — the RL gymnasium-style environment that ties all pipeline
components together.

Episode structure
─────────────────
reset(question)
    → observe initial state s_0

step(macro_action)     # called by PPOAgent
    → execute action via dispatcher
    → compute incremental reward
    → build next state s_{t+1}
    → return (s_{t+1}, reward, done, info)

Episode ends when:
  - macro_action == COMBINE  (agent chose to produce final answer), OR
  - step >= RL_MAX_STEPS     (budget exceeded)

The env caches all intermediate results (sub_questions, docs, sub_answers)
and passes them to the state builder and reward computer at every step.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Tuple, Any

import numpy as np

from core.config import Config
from rl.state   import build_state
from rl.actions import (
    MacroAction, dispatch,
    DecomposeResult, RetrieveResult, GenerateResult, CombineResult,
)
from rl.rewards import compute_all_rewards, combined_reward
import pipeline.decomposer as decomposer_mod
import pipeline.retriever  as retriever_mod
import pipeline.generator  as generator_mod
import pipeline.combiner   as combiner_mod

log = logging.getLogger(__name__)


class LegalQAEnv:
    """
    RL Environment for multi-hop legal question answering.

    Parameters
    ----------
    complexity_classifier : callable(question) → float
        Returns complexity confidence score ∈ [0,1].
        If None, defaults to 0.7 (assume complex).
    """

    def __init__(self, complexity_classifier=None):
        self.classifier = complexity_classifier
        self._reset_state()

    # ──────────────────────────────────────────────────────────────────────────
    # Environment API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        question: str
    ) -> np.ndarray:
        """
        Start a new episode.

        Returns the initial state vector s_0.
        """
        self._reset_state()
        self.question    = question.strip()


        # Get complexity score from classifier (or default)
        if self.classifier is not None:
            try:
                self.complexity_score = float(self.classifier(question))
            except Exception as e:
                log.warning("Classifier error: %s — defaulting to 0.7", e)
                self.complexity_score = 0.7
        else:
            self.complexity_score = 0.7

        log.info(
            "Env reset | question='%s…' | complexity=%.3f",
            question[:60], self.complexity_score,
        )
        return self._get_state()

    def get_action_mask(self) -> list:
        """
        Return a binary mask of length 4 enforcing valid step ordering.

        Valid sequence:
          step 0 → DECOMPOSE  [1, 0, 0, 0]
          step 1 → RETRIEVE   [0, 1, 0, 0]
          step 2 → GENERATE   [0, 0, 1, 0]
          step 3+ → COMBINE   [0, 0, 0, 1]
        """
        if self.step_count == 0:
            return [1, 0, 0, 0]
        elif self.step_count == 1:
            return [0, 1, 0, 0]
        elif self.step_count == 2:
            return [0, 0, 1, 0]
        else:
            return [0, 0, 0, 1]

    def step(
        self, macro_action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute macro_action and return (next_state, reward, done, info).

        Parameters
        ----------
        macro_action : int in {0,1,2,3}

        Returns
        -------
        next_state : np.ndarray (1538,)
        reward     : float
        done       : bool
        info       : dict with logs, action name, individual rewards
        """
        action_name = MacroAction.name_of(macro_action)
        log.info("Step %d: action=%s", self.step_count, action_name)

        # ── Dispatch ──
        result, log_msg = dispatch(
            macro_action,
            question         = self.question,
            sub_questions    = self.sub_questions,
            retrieve_result  = self.retrieve_result,
            generate_result  = self.generate_result,
            complexity_score = self.complexity_score,
            step             = self.step_count,
            decomposer       = decomposer_mod,
            retriever        = retriever_mod,
            generator        = generator_mod,
            combiner         = combiner_mod,
        )

        # ── Store result ──
        if isinstance(result, DecomposeResult):
            self.decompose_result = result
            self.sub_questions    = result.sub_questions

        elif isinstance(result, RetrieveResult):
            self.retrieve_result  = result
            self.doc_texts        = result.texts

        elif isinstance(result, GenerateResult):
            self.generate_result  = result
            self.sub_answers      = result.sub_answers

        elif isinstance(result, CombineResult):
            self.combine_result   = result
            self.final_answer     = result.final_answer

        # ── Step-wise shaped reward (dense signal for PPO) ──
        R = self._compute_step_reward(result)

        # ── Advance step ──
        self.step_count += 1
        self.actions_taken.append(action_name)
        self.cumulative_reward += R

        done = (
            isinstance(result, CombineResult)
            or self.step_count >= Config.RL_MAX_STEPS
        )

        next_state = self._get_state()

        # Individual rewards (for logging/snapshots, computed lazily)
        rewards_dict = compute_all_rewards(
            question     = self.question,
            final_answer = self.final_answer or " ".join(self.sub_answers),
            sub_questions= self.sub_questions,
            doc_texts    = self.doc_texts
        )

        info = {
            "action":             action_name,
            "log":                log_msg,
            "individual_rewards": rewards_dict,
            "step_reward":        R,
            "step":               self.step_count,
            "done":               done,
            "final_answer":       self.final_answer,
            "sub_questions":      self.sub_questions,
            "actions_taken":      self.actions_taken,
        }

        log.info(
            "Step %d done | reward=%.4f | done=%s | %s",
            self.step_count - 1, R, done, log_msg,
        )
        return next_state, R, done, info

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_step_reward(self, result) -> float:
        """
        Dense step-wise reward shaping.

        Instead of always calling combined_reward() (which requires a final
        answer to be non-empty), we give action-specific intermediate signals
        so PPO gets learning signal at every step:

          DECOMPOSE  → 0.2 × coverage_score
          RETRIEVE   → 0.3 × (at least one chunk retrieved)
          GENERATE   → 0.2 × answer_length_score
          COMBINE    → full combined_reward() on final answer
        """
        if isinstance(result, DecomposeResult):
            # Reward proportional to how well sub-questions cover the question
            coverage = getattr(result, "coverage_score", 0.0)
            R = 0.2 * float(coverage)
            log.debug("DECOMPOSE step reward: %.4f (coverage=%.3f)", R, coverage)
            return R

        elif isinstance(result, RetrieveResult):
            # Binary reward for retrieving at least one chunk
            num_chunks = len(result.chunks)
            R = 0.3 * float(num_chunks > 0)
            log.debug("RETRIEVE step reward: %.4f (chunks=%d)", R, num_chunks)
            return R

        elif isinstance(result, GenerateResult):
            # Reward based on non-empty, reasonably long answers
            answers   = result.sub_answers
            if not answers:
                return 0.0
            avg_len   = sum(len(a.split()) for a in answers) / len(answers)
            # Normalise: 0 words → 0, 50+ words → 1.0; clipped
            length_score = float(np.clip(avg_len / 50.0, 0.0, 1.0))
            R = 0.2 * length_score
            log.debug("GENERATE step reward: %.4f (avg_len=%.1f)", R, avg_len)
            return R

        elif isinstance(result, CombineResult):
            # Full combined reward at the terminal step
            rewards_dict = compute_all_rewards(
                question     = self.question,
                final_answer = result.final_answer,
                sub_questions= self.sub_questions,
                doc_texts    = self.doc_texts
            )
            R = combined_reward(rewards_dict)
            log.debug("COMBINE step reward: %.4f", R)
            return R

        return 0.0

    def _reset_state(self):
        self.question          = ""
        self.complexity_score  = 0.7
        self.step_count        = 0
        self.cumulative_reward = 0.0
        self.actions_taken:    List[str]         = []

        # Pipeline intermediate results
        self.decompose_result: Optional[DecomposeResult] = None
        self.retrieve_result:  Optional[RetrieveResult]  = None
        self.generate_result:  Optional[GenerateResult]  = None
        self.combine_result:   Optional[CombineResult]   = None

        # Flat lists for state builder
        self.sub_questions:    List[str]  = []
        self.doc_texts:        List[str]  = []
        self.sub_answers:      List[str]  = []
        self.final_answer:     str        = ""

    def _get_state(self) -> np.ndarray:
        return build_state(
            question         = self.question,
            sub_questions    = self.sub_questions,
            doc_texts        = self.doc_texts,
            sub_answers      = self.sub_answers,
            complexity_score = self.complexity_score,
            step             = self.step_count,
            max_steps        = Config.RL_MAX_STEPS,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Snapshot (for the chatbot API)
    # ──────────────────────────────────────────────────────────────────────────

    def get_result_snapshot(self) -> Dict[str, Any]:
        """Return a serialisable dict of the current episode results."""
        rewards = compute_all_rewards(
            question      = self.question,
            final_answer  = self.final_answer or " ".join(self.sub_answers),
            sub_questions = self.sub_questions,
            doc_texts     = self.doc_texts
        )
        retrieved_meta = []
        if self.retrieve_result:
            for c in self.retrieve_result.chunks:
                meta = c.get("metadata", {})
                retrieved_meta.append({
                    "article_num": meta.get("article_num", "?"),
                    "title":       meta.get("title", ""),
                    "text":        c.get("text", "")[:300],
                    "rerank_score": c.get("rerank_score", 0.0),
                })
        return {
            "question":          self.question,
            "complexity_score":  self.complexity_score,
            "sub_questions":     self.sub_questions,
            "retrieved_articles": retrieved_meta,
            "sub_answers":       self.sub_answers,
            "final_answer":      self.final_answer,
            "actions_taken":     self.actions_taken,
            "rewards":           rewards,
            "combined_reward":   combined_reward(rewards),
            "steps_taken":       self.step_count,
        }
