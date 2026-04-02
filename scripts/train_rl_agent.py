"""
scripts/train_rl_agent.py
--------------------------
PPO training loop for the RL-guided legal QA system.

Usage
─────
# From the project root:
python scripts/train_rl_agent.py

# With args:
python scripts/train_rl_agent.py --episodes 200 --dataset data/decompose_dataset.json

Training Data
─────────────
Uses decompose_dataset.json as the episode question pool.
Each episode = one complex question.
Gold answers are optional — if present in the dataset they improve
the correctness reward; otherwise, only retrieval/NLI/fluency rewards train.

Checkpointing
─────────────
Saves ppo_agent.pt to rl_model/ every SAVE_EVERY episodes.
Resume by re-running the script — it loads the checkpoint if it exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import List, Dict, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from rl.agent   import PPOAgent
from rl.env     import LegalQAEnv
from rl.actions import MacroAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_rl")


# ── Action schedule ───────────────────────────────────────────────────────────
# Ordered sequence of macro-actions that forms a complete episode:
# Decompose → Retrieve → Generate → Combine
EPISODE_ACTION_SEQUENCE = [
    MacroAction.DECOMPOSE,
    MacroAction.RETRIEVE,
    MacroAction.GENERATE,
    MacroAction.COMBINE,
]


# ── Classifier wrapper ────────────────────────────────────────────────────────

def load_classifier(device: str = "cpu"):
    """Load the trained LegalBERT complexity classifier."""
    classifier_path = Config.CLASSIFIER_MODEL_PATH
    if not os.path.exists(os.path.join(classifier_path, "best_model.pt")):
        log.warning(
            "Classifier not found at %s — using default complexity score 0.7",
            classifier_path,
        )
        return None

    import torch
    from transformers import AutoTokenizer
    # Import from project root
    sys.path.insert(0, Config.BASE_DIR)
    from classifier import LegalComplexityClassifier  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    model     = LegalComplexityClassifier(Config.CLASSIFIER_BERT_NAME)
    ckpt      = torch.load(
        os.path.join(classifier_path, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    import torch as _torch

    def score_fn(question: str) -> float:
        enc = tokenizer(
            question, return_tensors="pt",
            max_length=128, truncation=True, padding="max_length",
        )
        with _torch.no_grad():
            return model(enc["input_ids"], enc["attention_mask"]).item()

    log.info("Classifier loaded from %s", classifier_path)
    return score_fn


# ── Load training questions ───────────────────────────────────────────────────

def load_questions(dataset_path: str) -> List[Dict]:
    """Load questions from decompose_dataset.json."""
    if not os.path.exists(dataset_path):
        log.error("Dataset not found: %s", dataset_path)
        sys.exit(1)
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d items from %s", len(data), dataset_path)
    return data


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    episodes:    int  = Config.RL_EPOCHS,
    dataset:     str  = os.path.join(Config.BASE_DIR, "data", "decompose_dataset.json"),
    save_every:  int  = 20,
    log_every:   int  = 5,
    seed:        int  = 42,
    dry_run:     bool = False,
):
    random.seed(seed)
    np.random.seed(seed)

    data       = load_questions(dataset)
    classifier = load_classifier()
    env        = LegalQAEnv(complexity_classifier=classifier)
    agent      = PPOAgent()
    agent.load()   # resume from checkpoint if it exists

    episode_rewards: List[float] = []

    log.info("=" * 60)
    log.info("Starting PPO training | episodes=%d", episodes)
    log.info("=" * 60)

    for ep in range(1, episodes + 1):
        # Pick a random question each episode
        item      = random.choice(data)
        question  = item.get("complex_question", item.get("question", ""))
        gold      = None  # gold answers not in decompose_dataset by default

        if dry_run:
            log.info("[DRY RUN] Episode %d | Q: %s…", ep, question[:60])
            continue

        # ── Episode rollout ──
        state = env.reset(question, gold_answer=gold)
        ep_reward = 0.0

        # We follow the fixed action sequence: D→R→G→C
        # The agent ALSO sees the state so it can learn ordering preferences
        # In a future iteration, remove the fixed sequence to let the agent explore
        for action_id in EPISODE_ACTION_SEQUENCE:
            action, log_prob, value = agent.select_action(state)
            # For now: override with scheduled action to ensure valid episodes
            # (random exploration early; anneal to agent's choice as training progresses)
            scheduled_action = int(action_id)
            effective_action = (
                scheduled_action
                if ep <= episodes // 2         # first half: follow schedule
                else action                     # second half: trust the agent
            )

            next_state, reward, done, info = env.step(effective_action)

            agent.store(
                state    = state,
                action   = effective_action,
                reward   = reward,
                log_prob = log_prob,
                value    = value,
                done     = done,
            )

            ep_reward += reward
            state      = next_state

            if done:
                break

        # ── PPO update ──
        last_val = 0.0  # terminal state → value = 0
        metrics  = agent.update(last_value=last_val)
        episode_rewards.append(ep_reward)

        # ── Logging ──
        if ep % log_every == 0:
            avg_reward = np.mean(episode_rewards[-log_every:])
            log.info(
                "Ep %4d/%d | avg_R=%.4f | policy_loss=%.4f | "
                "value_loss=%.4f | entropy=%.4f",
                ep, episodes,
                avg_reward,
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["entropy"],
            )

        # ── Checkpoint ──
        if ep % save_every == 0:
            agent.save()
            log.info("Checkpoint saved at episode %d", ep)

    # Final save
    agent.save()
    log.info("Training complete. Final model saved.")
    log.info(
        "Mean reward over all episodes: %.4f",
        np.mean(episode_rewards) if episode_rewards else 0.0,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for Legal QA")
    parser.add_argument("--episodes",   type=int, default=Config.RL_EPOCHS)
    parser.add_argument("--dataset",    type=str,
                        default=os.path.join(Config.BASE_DIR, "data", "decompose_dataset.json"))
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every",  type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run without actual LLM calls (structure test only)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes   = args.episodes,
        dataset    = args.dataset,
        save_every = args.save_every,
        log_every  = args.log_every,
        seed       = args.seed,
        dry_run    = args.dry_run,
    )
