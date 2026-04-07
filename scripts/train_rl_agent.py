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

Training Phases
───────────────
Phase 1 — Warm-start (0 … warmup_frac of total episodes)
  Force the correct D→R→G→C sequence every step.
  PPO sees good trajectories immediately (reward ~0.5–0.8).

Phase 2 — Mixed (warmup_frac … mixed_frac)
  50 / 50 split each episode: either follow fixed schedule OR use the
  agent's masked policy.  Smooth transition to autonomous control.

Phase 3 — Full RL (mixed_frac … end)
  Agent's masked policy used exclusively.
  Action masking still enforces D→R→G→C ordering to prevent crashes.

Checkpointing
─────────────
Saves ppo_agent.pt  to rl_model/ every --save-every episodes.
Saves training_metrics.json + training_curves.png at the end.
Resume by re-running — loads the checkpoint if present.
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

# ── Project root on path ──────────────────────────────────────────────────────
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


# ── Fixed action schedule (warm-start / scheduled sequence) ───────────────────
# One complete episode: DECOMPOSE(0) → RETRIEVE(1) → GENERATE(2) → COMBINE(3)
FIXED_SEQUENCE = [
    int(MacroAction.DECOMPOSE),
    int(MacroAction.RETRIEVE),
    int(MacroAction.GENERATE),
    int(MacroAction.COMBINE),
]


# ── Classifier loader ─────────────────────────────────────────────────────────

def load_classifier(device: str = "cpu"):
    """Load the trained LegalBERT complexity classifier (optional)."""
    classifier_path = Config.CLASSIFIER_MODEL_PATH
    if not os.path.exists(os.path.join(classifier_path, "best_model.pt")):
        log.warning(
            "Classifier not found at %s — using default complexity score 0.7",
            classifier_path,
        )
        return None

    import torch
    from transformers import AutoTokenizer
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


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_questions(dataset_path: str) -> List[Dict]:
    """Load questions from decompose_dataset.json."""
    if not os.path.exists(dataset_path):
        log.error("Dataset not found: %s", dataset_path)
        sys.exit(1)
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d items from %s", len(data), dataset_path)
    return data


# ── Phase helper ──────────────────────────────────────────────────────────────

def _get_phase(ep: int, total: int, warmup_frac: float, mixed_frac: float) -> str:
    """
    Return current training phase label.

    Phase 1 — 'warmup' : forced schedule
    Phase 2 — 'mixed'  : 50/50 schedule vs agent
    Phase 3 — 'full_rl': agent only
    """
    progress = ep / max(total, 1)
    if progress < warmup_frac:
        return "warmup"
    elif progress < mixed_frac:
        return "mixed"
    return "full_rl"


# ── Action selection per phase ────────────────────────────────────────────────

def _select_action(
    phase:    str,
    step_idx: int,
    state:    np.ndarray,
    mask:     list,
    agent:    PPOAgent,
) -> tuple:
    """
    Return (effective_action, log_prob, value) according to current phase.

    Phase 'warmup' → always follow FIXED_SEQUENCE (D→R→G→C).
    Phase 'mixed'  → 50 % schedule, 50 % agent's masked policy.
    Phase 'full_rl'→ agent's masked policy only.

    In all cases action_mask is respected.
    """
    scheduled_action = FIXED_SEQUENCE[step_idx]

    if phase == "warmup":
        # Bootstrap PPO with perfect demonstrations
        # We still call forward() to get log_prob + value for the buffer
        state_t = _state_tensor(state, agent)
        import torch
        from torch.distributions import Categorical
        with torch.no_grad():
            logits, value = agent.network(state_t)
            mask_t = torch.FloatTensor(mask).to(agent.device)
            logits = logits + (1.0 - mask_t) * -1e9
            dist   = Categorical(logits=logits)
            act_t  = torch.tensor(scheduled_action).to(agent.device)
            log_p  = dist.log_prob(act_t)
        return scheduled_action, log_p.item(), value.squeeze(-1).item()

    elif phase == "mixed":
        if random.random() < 0.5:
            # Follow schedule half the time
            state_t = _state_tensor(state, agent)
            import torch
            from torch.distributions import Categorical
            with torch.no_grad():
                logits, value = agent.network(state_t)
                mask_t = torch.FloatTensor(mask).to(agent.device)
                logits = logits + (1.0 - mask_t) * -1e9
                dist   = Categorical(logits=logits)
                act_t  = torch.tensor(scheduled_action).to(agent.device)
                log_p  = dist.log_prob(act_t)
            return scheduled_action, log_p.item(), value.squeeze(-1).item()
        else:
            # Agent's masked policy
            return agent.select_action(state, action_mask=mask)

    else:  # full_rl
        return agent.select_action(state, action_mask=mask)


def _state_tensor(state: np.ndarray, agent: PPOAgent):
    import torch
    return torch.FloatTensor(state).unsqueeze(0).to(agent.device)


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    episodes:         int   = Config.RL_EPOCHS,
    dataset:          str   = os.path.join(Config.BASE_DIR, "data", "decompose_dataset.json"),
    save_every:       int   = 20,
    log_every:        int   = 5,
    seed:             int   = 42,
    warmup_frac:      float = 0.33,
    mixed_frac:       float = 0.66,
    dry_run:          bool  = False,
    checkpoint_name:  str   = "ppo_agent_100",   # filename without .pt
):
    random.seed(seed)
    np.random.seed(seed)

    data       = load_questions(dataset)
    classifier = load_classifier()
    env        = LegalQAEnv(complexity_classifier=classifier)
    agent      = PPOAgent()
    agent.load()  # resume checkpoint if exists

    # Resolve checkpoint file path
    ckpt_path = os.path.join(Config.RL_MODEL_PATH, f"{checkpoint_name}.pt")
    log.info("Checkpoint will be saved to: %s", ckpt_path)

    # ── Stats trackers ────────────────────────────────────────────────────────
    episode_rewards:  List[float] = []
    policy_losses:    List[float] = []
    value_losses:     List[float] = []
    entropies:        List[float] = []
    action_counts:    Dict[str, int] = {a.name: 0 for a in MacroAction}

    log.info("=" * 65)
    log.info("Starting 3-Phase PPO Training | episodes=%d", episodes)
    log.info(
        "  Phase 1 (warm-start) : eps 1 – %d",
        int(episodes * warmup_frac),
    )
    log.info(
        "  Phase 2 (mixed)      : eps %d – %d",
        int(episodes * warmup_frac) + 1,
        int(episodes * mixed_frac),
    )
    log.info(
        "  Phase 3 (full RL)    : eps %d – %d",
        int(episodes * mixed_frac) + 1,
        episodes,
    )
    log.info("=" * 65)

    for ep in range(1, episodes + 1):
        # ── Pick question ──────────────────────────────────────────────────
        item     = random.choice(data)
        question = item.get("complex_question", item.get("question", ""))
        gold     = item.get("answer", None)   # optional gold answer

        phase = _get_phase(ep, episodes, warmup_frac, mixed_frac)

        if dry_run:
            log.info(
                "[DRY RUN] Ep %d | phase=%s | Q: %s…",
                ep, phase, question[:60],
            )
            continue

        # ── Episode rollout ────────────────────────────────────────────────
        state     = env.reset(question, gold_answer=gold)
        ep_reward = 0.0

        for step_idx in range(len(FIXED_SEQUENCE)):
            # Get mask BEFORE the step (based on current step_count)
            mask = env.get_action_mask()

            # Select action according to current phase
            effective_action, log_prob, value = _select_action(
                phase, step_idx, state, mask, agent
            )

            # Execute in environment
            next_state, reward, done, info = env.step(effective_action)

            # Store (always use effective_action — no mismatch)
            agent.store(
                state    = state,
                action   = effective_action,
                reward   = reward,
                log_prob = log_prob,
                value    = value,
                done     = done,
            )

            # Track action distribution
            action_name = MacroAction.name_of(effective_action)
            action_counts[action_name] += 1

            ep_reward += reward
            state      = next_state

            if done:
                break

        # ── PPO update ─────────────────────────────────────────────────────
        metrics = agent.update(last_value=0.0)
        episode_rewards.append(ep_reward)
        policy_losses.append(metrics["policy_loss"])
        value_losses.append(metrics["value_loss"])
        entropies.append(metrics["entropy"])

        # ── Per-episode logging ────────────────────────────────────────────
        if ep % log_every == 0:
            window = min(log_every, len(episode_rewards))
            avg_R  = float(np.mean(episode_rewards[-window:]))
            log.info(
                "Ep %4d/%d | phase=%-8s | avg_R=%+.4f | "
                "π_loss=%+.4f | V_loss=%.4f | H=%.4f",
                ep, episodes, phase,
                avg_R,
                metrics["policy_loss"],
                metrics["value_loss"],
                metrics["entropy"],
            )
            log.info(
                "  Actions this episode: %s | subQs=%d | chunks=%d",
                " → ".join(info.get("actions_taken", [])),
                len(info.get("sub_questions", [])),
                len(env.doc_texts),
            )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if ep % save_every == 0:
            agent.save()
            log.info("Checkpoint saved at episode %d", ep)

    # ── Final save ─────────────────────────────────────────────────────────────
    agent.save()
    log.info("Training complete. Model saved.")
    log.info(
        "Mean reward (all episodes): %.4f",
        np.mean(episode_rewards) if episode_rewards else 0.0,
    )

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    os.makedirs(Config.RL_MODEL_PATH, exist_ok=True)
    metrics_path = os.path.join(Config.RL_MODEL_PATH, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "episode_rewards": episode_rewards,
            "policy_losses":   policy_losses,
            "value_losses":    value_losses,
            "entropies":       entropies,
            "action_counts":   action_counts,
        }, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # ── Plot training curves (if matplotlib available) ─────────────────────────
    _plot_curves(episode_rewards, policy_losses, value_losses, entropies)


def _plot_curves(
    rewards:       List[float],
    policy_losses: List[float],
    value_losses:  List[float],
    entropies:     List[float],
):
    """Save training_curves.png to rl_model/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("PPO Training Curves — Legal QA RL Agent", fontsize=14)

        def _smooth(vals, k=10):
            if len(vals) < k:
                return vals
            return np.convolve(vals, np.ones(k) / k, mode="valid").tolist()

        axes[0, 0].plot(_smooth(rewards), color="steelblue")
        axes[0, 0].set_title("Episode Reward (smoothed)")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")

        axes[0, 1].plot(_smooth(policy_losses), color="tomato")
        axes[0, 1].set_title("Policy Loss (smoothed)")
        axes[0, 1].set_xlabel("Episode")

        axes[1, 0].plot(_smooth(value_losses), color="seagreen")
        axes[1, 0].set_title("Value Loss (smoothed)")
        axes[1, 0].set_xlabel("Episode")

        axes[1, 1].plot(_smooth(entropies), color="goldenrod")
        axes[1, 1].set_title("Entropy (smoothed)")
        axes[1, 1].set_xlabel("Episode")

        plt.tight_layout()
        out_path = os.path.join(Config.RL_MODEL_PATH, "training_curves.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        log.info("Training curves saved → %s", out_path)
    except ImportError:
        log.warning("matplotlib not installed — skipping training curves plot.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for Legal QA")
    parser.add_argument("--episodes",     type=int,   default=Config.RL_EPOCHS)
    parser.add_argument("--dataset",      type=str,
                        default=os.path.join(Config.BASE_DIR, "data", "decompose_dataset.json"))
    parser.add_argument("--save-every",   type=int,   default=20)
    parser.add_argument("--log-every",    type=int,   default=5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--checkpoint-name", type=str, default="ppo_agent",
                        help="Filename (without .pt) for the saved checkpoint, e.g. ppo_ep50")
    parser.add_argument("--warmup-frac",  type=float, default=0.33,
                        help="Fraction of episodes for Phase 1 warm-start (default 0.33)")
    parser.add_argument("--mixed-frac",   type=float, default=0.66,
                        help="Fraction of episodes up to which Phase 2 runs (default 0.66)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Run without actual LLM calls — structure test only")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        episodes         = args.episodes,
        dataset          = args.dataset,
        save_every       = args.save_every,
        log_every        = args.log_every,
        seed             = args.seed,
        warmup_frac      = args.warmup_frac,
        mixed_frac       = args.mixed_frac,
        dry_run          = args.dry_run,
        checkpoint_name  = args.checkpoint_name,
    )
