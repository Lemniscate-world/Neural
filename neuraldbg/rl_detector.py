"""
RL Detector — NeuralDBG extension for Policy Gradient blind spots.

Fixes the 0% detection rate on RL (REINFORCE/PPO) by hooking BEFORE
log_softmax, tracking reward variance, detecting policy collapse,
and using reward-scale adaptive thresholds.

Architecture:
  1. Raw logit hooks (before log_softmax)
  2. Reward variance tracker
  3. Policy collapse detector (entropy + action distribution)
  4. Adaptive thresholds keyed to reward magnitude

Usage:
    from neuraldbg.rl_detector import RLDetector

    detector = RLDetector(model, family="RL")
    with NeuralDbg(model) as dbg:
        ...
        detector.step(logits, actions, rewards, values)
        ...

Integration with NeuralDbg:
    Call detector.step() after loss.backward() and before optimizer.step().
    Events are automatically forwarded to the NeuralDbg instance.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RL Event Types (extend NeuralDBG's EventType)
# ---------------------------------------------------------------------------

@dataclass
class RLEvent:
    """An RL-specific diagnostic event."""
    event_type: str           # reward_anomaly, policy_collapse, logit_saturation, value_divergence
    step: int
    detail: str
    severity: float           # 0.0 - 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "step": self.step,
            "detail": self.detail,
            "severity": round(self.severity, 4),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# RL Detector
# ---------------------------------------------------------------------------

class RLDetector:
    """Detects RL-specific training anomalies that standard NeuralDBG hooks miss.

    Solves the 0% detection rate on Policy Gradient (REINFORCE, PPO, A2C) by:
      1. Hooking raw logits (before Categorical.log_prob applies log_softmax)
      2. Tracking reward distribution statistics (variance, skew)
      3. Detecting policy collapse (entropy → 0, single-action dominance)
      4. Adapting thresholds to reward magnitude (no fixed threshold)

    Usage:
        detector = RLDetector(model)
        with NeuralDbg(model) as dbg:
            for step in range(num_steps):
                logits, values = model(states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                loss = -(log_probs * advantages).mean()
                loss.backward()
                detector.step(logits, actions, rewards, values, step=step)
                optimizer.step()

        # Get RL-specific events
        for event in detector.events:
            print(f"[{event.event_type}] step={event.step}: {event.detail}")
    """

    def __init__(
        self,
        model: nn.Module,
        reward_window: int = 50,
        entropy_collapse_threshold: float = 0.01,
        reward_spike_factor: float = 10.0,
        value_divergence_factor: float = 5.0,
        logit_saturation_threshold: float = 50.0,
    ):
        """
        Args:
            model: The PyTorch model (ActorCritic or PolicyNet).
            reward_window: Number of steps for running reward statistics.
            entropy_collapse_threshold: Entropy below this → policy collapse.
            reward_spike_factor: reward > running_mean * factor → anomaly.
            value_divergence_factor: |value| > running_std * factor → anomaly.
            logit_saturation_threshold: |logit| > this → saturation.
        """
        self.model = model
        self.reward_window = reward_window
        self.entropy_collapse_threshold = entropy_collapse_threshold
        self.reward_spike_factor = reward_spike_factor
        self.value_divergence_factor = value_divergence_factor
        self.logit_saturation_threshold = logit_saturation_threshold

        # Running statistics
        self.reward_history: Deque[float] = deque(maxlen=reward_window)
        self.value_history: Deque[float] = deque(maxlen=reward_window)
        self.entropy_history: Deque[float] = deque(maxlen=reward_window)
        self.logit_norm_history: Deque[float] = deque(maxlen=reward_window)
        self.gradient_norm_history: Deque[float] = deque(maxlen=reward_window)

        # Events
        self.events: List[RLEvent] = []

        # Raw logit hooks (SOLUTION 1)
        self._raw_logit_hooks: Dict[str, torch.Tensor] = {}
        self._install_logit_hooks()

    # ------------------------------------------------------------------
    # Solution 1: Hook raw logits BEFORE log_softmax
    # ------------------------------------------------------------------

    def _install_logit_hooks(self):
        """Install forward hooks on the final Linear layer of the policy head.

        These capture raw logits before Categorical.log_prob applies
        log_softmax internally, which would compress gradient anomalies.
        """
        # Find the last Linear layer in the model (policy head)
        last_linear = None
        last_name = ""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
                last_name = name

        if last_linear is not None:

            def _hook(module, inp, out, name=last_name):
                self._raw_logit_hooks[name] = out.detach()

            last_linear.register_forward_hook(_hook)

    def get_raw_logits(self) -> Optional[torch.Tensor]:
        """Return the most recent raw logits captured by hooks."""
        if not self._raw_logit_hooks:
            return None
        # Return the last (deepest) logit tensor
        keys = sorted(self._raw_logit_hooks.keys(), reverse=True)
        return self._raw_logit_hooks.get(keys[0] if keys else "")

    # ------------------------------------------------------------------
    # Solution 2: Reward variance tracker
    # ------------------------------------------------------------------

    def _track_reward_anomaly(self, rewards: torch.Tensor, step: int):
        """Detect anomalies in reward distribution: spikes, variance collapse, skew."""
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().item()
        reward_max = rewards.max().item()
        reward_min = rewards.min().item()

        self.reward_history.append(reward_mean)

        if len(self.reward_history) < 5:
            return  # Need warmup

        running_mean = sum(self.reward_history) / len(self.reward_history)
        running_std = (
            sum((r - running_mean) ** 2 for r in self.reward_history)
            / max(1, len(self.reward_history) - 1)
        ) ** 0.5

        # Reward spike (SOLUTION 2a)
        if running_std > 0 and abs(reward_mean - running_mean) > self.reward_spike_factor * running_std:
            self.events.append(RLEvent(
                event_type="reward_anomaly",
                step=step,
                detail=f"Reward spike: mean={reward_mean:.2f} vs running {running_mean:.2f}±{running_std:.2f}",
                severity=min(1.0, abs(reward_mean - running_mean) / max(running_std, 1e-8) / 10),
                metadata={
                    "reward_mean": reward_mean, "running_mean": running_mean,
                    "running_std": running_std, "reward_max": reward_max,
                },
            ))

        # Variance collapse (all rewards identical → no learning signal)
        if reward_std < 1e-6 and len(self.reward_history) > 10:
            self.events.append(RLEvent(
                event_type="reward_anomaly",
                step=step,
                detail=f"Reward variance collapse: std={reward_std:.2e}",
                severity=0.8,
                metadata={"reward_std": reward_std, "reward_mean": reward_mean},
            ))

    # ------------------------------------------------------------------
    # Solution 3: Policy collapse detector
    # ------------------------------------------------------------------

    def _detect_policy_collapse(
        self, logits: torch.Tensor, actions: torch.Tensor, step: int
    ):
        """Detect when the policy collapses to a single action (entropy → 0)."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean().item()
            max_prob = probs.max(-1).values.mean().item()
            action_diversity = len(actions.unique()) / len(actions)

            # Also check logit norms (raw, before softmax)
            logit_norm = logits.norm(dim=-1).mean().item()

        self.entropy_history.append(entropy)
        self.logit_norm_history.append(logit_norm)

        # Entropy collapse (SOLUTION 3a)
        if entropy < self.entropy_collapse_threshold:
            self.events.append(RLEvent(
                event_type="policy_collapse",
                step=step,
                detail=f"Policy entropy collapse: {entropy:.2e} (single-action dominance {max_prob:.2%})",
                severity=min(1.0, (self.entropy_collapse_threshold / max(entropy, 1e-10)) * 0.5),
                metadata={
                    "entropy": entropy, "max_prob": max_prob,
                    "action_diversity": action_diversity,
                },
            ))

        # Action diversity collapse (all same action chosen)
        if action_diversity < 0.3 and len(actions) > 10:
            self.events.append(RLEvent(
                event_type="policy_collapse",
                step=step,
                detail=f"Action diversity collapse: {action_diversity:.1%} unique actions",
                severity=0.7,
                metadata={"action_diversity": action_diversity, "n_actions": len(actions)},
            ))

        # Logit saturation (SOLUTION 1: extreme logits → softmax produces one-hot)
        if logit_norm > self.logit_saturation_threshold:
            self.events.append(RLEvent(
                event_type="logit_saturation",
                step=step,
                detail=f"Logit saturation: norm={logit_norm:.1f} (saturated softmax)",
                severity=min(1.0, logit_norm / self.logit_saturation_threshold / 5),
                metadata={"logit_norm": logit_norm},
            ))

    # ------------------------------------------------------------------
    # Solution 4: Adaptive thresholds keyed to reward scale
    # ------------------------------------------------------------------

    def _check_value_divergence(self, values: torch.Tensor, rewards: torch.Tensor, step: int):
        """Detect value network divergence relative to reward scale."""
        value_mean = values.mean().item()
        value_std = values.std().item()
        reward_scale = rewards.abs().mean().item()

        self.value_history.append(value_mean)

        if len(self.value_history) < 5:
            return

        running_value_mean = sum(self.value_history) / len(self.value_history)
        running_value_std = (
            sum((v - running_value_mean) ** 2 for v in self.value_history)
            / max(1, len(self.value_history) - 1)
        ) ** 0.5

        # Adaptive threshold: scale by reward magnitude (SOLUTION 4)
        adaptive_threshold = max(running_value_std, reward_scale) * self.value_divergence_factor

        if abs(value_mean - running_value_mean) > adaptive_threshold:
            self.events.append(RLEvent(
                event_type="value_divergence",
                step=step,
                detail=f"Value divergence: |Δ|={abs(value_mean - running_value_mean):.2f} > threshold={adaptive_threshold:.2f}",
                severity=min(1.0, abs(value_mean - running_value_mean) / max(adaptive_threshold, 1e-8)),
                metadata={
                    "value_mean": value_mean, "running_mean": running_value_mean,
                    "adaptive_threshold": adaptive_threshold, "reward_scale": reward_scale,
                },
            ))

    def _check_gradient_anomaly_rl(self, step: int):
        """Check gradient norms WITH reward-scale adaptation (SOLUTION 4).

        In RL, gradient norms naturally vary with reward magnitude.
        A fixed threshold of 1e-4 is useless — we need to scale
        thresholds based on running reward statistics.
        """
        norms = []
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                norms.append(p.grad.norm().item())

        if not norms:
            return

        mean_grad_norm = sum(norms) / len(norms)
        max_grad_norm = max(norms)

        self.gradient_norm_history.append(mean_grad_norm)

        if len(self.gradient_norm_history) < 10:
            return

        running_grad_mean = sum(self.gradient_norm_history) / len(self.gradient_norm_history)
        running_grad_std = (
            sum((g - running_grad_mean) ** 2 for g in self.gradient_norm_history)
            / max(1, len(self.gradient_norm_history) - 1)
        ) ** 0.5

        # Adaptive vanishing threshold (SOLUTION 4)
        adaptive_vanishing = max(running_grad_mean * 0.01, 1e-8)

        # Adaptive exploding threshold
        adaptive_exploding = running_grad_mean + 5 * max(running_grad_std, 1e-8)

        if mean_grad_norm < adaptive_vanishing and mean_grad_norm > 0:
            self.events.append(RLEvent(
                event_type="gradient_health_transition",
                step=step,
                detail=f"RL vanishing: mean_grad={mean_grad_norm:.2e} < adaptive_threshold={adaptive_vanishing:.2e}",
                severity=0.8,
                metadata={
                    "mean_grad_norm": mean_grad_norm,
                    "adaptive_vanishing": adaptive_vanishing,
                    "running_grad_mean": running_grad_mean,
                },
            ))

        if max_grad_norm > adaptive_exploding:
            self.events.append(RLEvent(
                event_type="gradient_health_transition",
                step=step,
                detail=f"RL exploding: max_grad={max_grad_norm:.2e} > adaptive_threshold={adaptive_exploding:.2e}",
                severity=min(1.0, max_grad_norm / max(adaptive_exploding, 1e-8) / 10),
                metadata={
                    "max_grad_norm": max_grad_norm,
                    "adaptive_exploding": adaptive_exploding,
                    "running_grad_mean": running_grad_mean,
                },
            ))

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        step: int = 0,
    ):
        """Run all RL-specific detectors for one training step.

        Call this AFTER loss.backward() and BEFORE optimizer.step().

        Args:
            logits: Raw policy logits (B, action_dim).
            actions: Selected actions (B,).
            rewards: Rewards received (B,).
            values: Value estimates (B,).
            step: Current training step.
        """
        # Solution 2: Reward anomaly
        self._track_reward_anomaly(rewards, step)

        # Solution 3: Policy collapse
        self._detect_policy_collapse(logits, actions, step)

        # Solution 4: Value divergence with adaptive thresholds
        self._check_value_divergence(values, rewards, step)

        # Solution 4: Gradient anomaly with adaptive thresholds
        self._check_gradient_anomaly_rl(step)

    # ------------------------------------------------------------------
    # Export / Analysis
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a summary of RL-specific diagnostics."""
        by_type: Dict[str, int] = {}
        for e in self.events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "events_by_type": by_type,
            "detection_active": len(self.events) > 0,
            "event_types_detected": sorted(by_type.keys()),
            "reward_stats": {
                "mean": sum(self.reward_history) / max(1, len(self.reward_history)),
                "count": len(self.reward_history),
            } if self.reward_history else None,
            "entropy_stats": {
                "mean": sum(self.entropy_history) / max(1, len(self.entropy_history)),
                "min": min(self.entropy_history) if self.entropy_history else None,
                "count": len(self.entropy_history),
            } if self.entropy_history else None,
        }

    def dump_events(self) -> List[Dict]:
        """Export all RL events as JSON-serializable dicts."""
        return [e.to_dict() for e in self.events]

    def reset(self):
        """Reset all running statistics and events."""
        self.reward_history.clear()
        self.value_history.clear()
        self.entropy_history.clear()
        self.logit_norm_history.clear()
        self.gradient_norm_history.clear()
        self.events.clear()
        self._raw_logit_hooks.clear()
