"""
rl/agent.py
-----------
PPO (Proximal Policy Optimization) agent for the Legal QA RL environment.

Network Architecture
────────────────────
Input  : state vector s_t  ∈ ℝ^1538
Shared : Linear(1538→512) → ReLU → Linear(512→256) → ReLU
Policy : Linear(256→4) → Softmax    [macro-action probs]
Value  : Linear(256→1)              [state value V(s)]

Training uses:
  - PPO clip objective  (clip_eps = 0.2)
  - Generalised Advantage Estimation (GAE, λ=0.95)
  - Value-function MSE loss
  - Entropy bonus for exploration
"""

from __future__ import annotations

import os
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core.config import Config

log = logging.getLogger(__name__)


# ── Policy + Value Network ────────────────────────────────────────────────────

class PPONetwork(nn.Module):
    """
    Shared backbone with separate policy (actor) and value (critic) heads.
    """
    def __init__(
        self,
        state_dim:  int = Config.RL_STATE_DIM,
        action_dim: int = Config.RL_ACTION_DIM,
        hidden_dim: int = Config.RL_HIDDEN_DIM,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)   # logits
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, state_value)."""
        feat  = self.backbone(state)
        logits = self.policy_head(feat)
        value  = self.value_head(feat).squeeze(-1)
        return logits, value

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Returns
        -------
        action   : int
        log_prob : scalar tensor
        value    : scalar tensor
        """
        logits, value = self(state)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        log_prob      = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate old actions under the current policy (for PPO update).

        Returns
        -------
        log_probs : (batch,)
        values    : (batch,)
        entropy   : scalar
        """
        logits, values = self(states)
        dist           = Categorical(logits=logits)
        log_probs      = dist.log_prob(actions)
        entropy        = dist.entropy().mean()
        return log_probs, values, entropy


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Collects one episode of experience."""
    def __init__(self):
        self.states:   List[np.ndarray] = []
        self.actions:  List[int]        = []
        self.rewards:  List[float]      = []
        self.log_probs:List[float]      = []
        self.values:   List[float]      = []
        self.dones:    List[bool]       = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = Config.RL_GAMMA,
        lam: float   = Config.RL_GAE_LAMBDA,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discounted returns and GAE advantages.

        Returns
        -------
        returns    : (T,) discounted returns
        advantages : (T,) GAE advantages (normalised)
        """
        T         = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae  = 0.0
        values    = self.values + [last_value]

        for t in reversed(range(T)):
            mask     = 0.0 if self.dones[t] else 1.0
            delta    = self.rewards[t] + gamma * values[t+1] * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        # Normalise advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO agent wrapping PPONetwork with training utilities.

    Usage
    -----
    agent = PPOAgent()
    action = agent.select_action(state)   # during rollout
    agent.update(buffer)                   # after each episode
    agent.save() / agent.load()           # checkpointing
    """

    def __init__(
        self,
        state_dim:  int   = Config.RL_STATE_DIM,
        action_dim: int   = Config.RL_ACTION_DIM,
        hidden_dim: int   = Config.RL_HIDDEN_DIM,
        lr:         float = Config.RL_LR,
        clip_eps:   float = Config.RL_CLIP_EPS,
        entropy_coef: float = 0.01,
        value_coef:   float = 0.5,
        ppo_epochs:   int   = 4,     # inner update epochs per rollout
        device: Optional[str] = None,
    ):
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.ppo_epochs   = ppo_epochs

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        log.info("PPOAgent device: %s", self.device)

        self.network   = PPONetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[list] = None,
    ) -> Tuple[int, float, float]:
        """
        Select action for one step (inference mode).

        Parameters
        ----------
        state       : current state vector
        action_mask : optional length-4 binary list.  Positions with 0 are
                      masked out (logit set to -1e9) so they are never sampled.

        Returns
        -------
        (action_id, log_prob_scalar, value_scalar)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.network(state_t)

            # Apply mask: invalid actions get logit = -1e9
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).to(self.device)  # (4,)
                logits = logits + (1.0 - mask_t) * -1e9

            dist     = Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze(-1).item()

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ):
        """Store one step in the rollout buffer."""
        self.buffer.add(state, action, reward, log_prob, value, done)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Run PPO update over the current buffer contents.
        Call at the end of each episode.

        Returns dict with training metrics.
        """
        returns, advantages = self.buffer.compute_returns_and_advantages(last_value)

        # Tensors
        states_t    = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions_t   = torch.LongTensor(self.buffer.actions).to(self.device)
        old_lp_t    = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns_t   = torch.FloatTensor(returns).to(self.device)
        adv_t       = torch.FloatTensor(advantages).to(self.device)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        for _ in range(self.ppo_epochs):
            log_probs, values, entropy = self.network.evaluate(states_t, actions_t)

            # Ratio r(θ) = π_θ(a|s) / π_θ_old(a|s)
            ratios = torch.exp(log_probs - old_lp_t.detach())

            # Surrogate losses
            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss  = nn.MSELoss()(values, returns_t)

            # Total loss
            loss = (
                policy_loss
                + self.value_coef  * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()

        e = self.ppo_epochs
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / e,
            "value_loss":  total_value_loss  / e,
            "entropy":     total_entropy     / e,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Save checkpoint.

        path can be:
          - None          → saves to Config.RL_MODEL_PATH/ppo_agent.pt
          - a directory   → saves to <path>/ppo_agent.pt
          - a .pt file    → saves directly to that file
        """
        if path is None:
            dir_path  = Config.RL_MODEL_PATH
            ckpt_file = os.path.join(dir_path, "ppo_agent.pt")
        elif path.endswith(".pt"):
            dir_path  = os.path.dirname(path) or Config.RL_MODEL_PATH
            ckpt_file = path
        else:
            dir_path  = path
            ckpt_file = os.path.join(path, "ppo_agent.pt")

        os.makedirs(dir_path, exist_ok=True)
        torch.save({
            "network_state":   self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, ckpt_file)
        log.info("PPOAgent saved → %s", ckpt_file)
        return ckpt_file

    def load(self, path: Optional[str] = None) -> None:
        dir_path  = path or Config.RL_MODEL_PATH
        ckpt_file = os.path.join(dir_path, Config.RL_MODEL_FILENAME)
        if not os.path.exists(ckpt_file):
            log.warning("No checkpoint found at %s — starting fresh.", ckpt_file)
            return
        ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        log.info("PPOAgent loaded ← %s", ckpt_file)
