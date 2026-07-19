#!/usr/bin/env python3
"""
RL (PPO-style / REINFORCE) failure scenarios with NeuralDBG + RLDetector.

Covers: reward hacking, policy collapse, value explosion, logit saturation.
Uses the RLDetector module to solve the 0% detection blind spot via:
  1. Raw logit hooks (before log_softmax)
  2. Reward variance tracking
  3. Policy collapse detection (entropy + action diversity)
  4. Adaptive thresholds keyed to reward magnitude

Usage:
    python examples/demo_rl_failures.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from neuraldbg import NeuralDbg
from neuraldbg.rl_detector import RLDetector


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.policy = PolicyNet(state_dim, action_dim, hidden)
        self.value = ValueNet(state_dim, hidden)

    def forward(self, state):
        logits = self.policy(state)
        value = self.value(state)
        return logits, value


def _make_dummy_batch(state_dim=8, action_dim=4, batch_size=16):
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)
    return states, actions, rewards, dones


def _compute_advantages(rewards, values, gamma=0.99):
    with torch.no_grad():
        advantages = rewards - values
    return advantages


def train_ppo(model, num_steps=20, lr=3e-4, state_dim=8, action_dim=4,
              reward_scale=1.0, use_rl_detector=True):
    """Train an Actor-Critic with PPO-style loss and RLDetector integration."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    # Initialize RL detector (solves the 0% detection blind spot)
    rl_detector = RLDetector(model) if use_rl_detector else None

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0,
                   family="RL") as dbg:
        for step in range(num_steps):
            optimizer.zero_grad()
            dbg.step = step

            states, actions, rewards, dones = _make_dummy_batch(state_dim, action_dim)
            rewards = rewards * reward_scale
            logits, values = model(states)

            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            advantages = _compute_advantages(rewards, values, gamma)
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = ((values - rewards) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            loss.backward()
            dbg.record_loss(loss.item())

            # RLDetector: check logits, actions, rewards, values AFTER backward
            if rl_detector is not None:
                rl_detector.step(logits, actions, rewards, values, step=step)

            optimizer.step()

    # Merge RL detector events into NeuralDBG events
    if rl_detector is not None:
        for rl_event in rl_detector.dump_events():
            # Create an event-like object compatible with NeuralDBG's Event protocol
            merged = type('RLEvent', (), {
                'event_type': rl_event['event_type'],
                'step': rl_event['step'],
                'detail': rl_event['detail'],
                'severity': rl_event['severity'],
                'confidence': rl_event['severity'],  # severity acts as confidence
                'metadata': rl_event['metadata'],
                'layer_name': 'rl_policy',      # RL events are policy-level
                'from_state': 'healthy',
                'to_state': rl_event['event_type'],
                'to_dict': lambda self, d=rl_event: d,
            })()
            dbg.events.append(merged)

    return dbg, rl_detector


def analyze_results(dbg, rl_detector=None):
    result = {
        "events": dbg.events,
    }
    try:
        result["couplings"] = dbg.detect_coupled_failures()
    except Exception:
        result["couplings"] = []
    try:
        result["mermaid"] = dbg.export_mermaid_causal_graph()
    except Exception:
        result["mermaid"] = ""
    try:
        result["hypotheses"] = dbg.explain_failure("vanishing_gradients")
    except Exception:
        result["hypotheses"] = []
    try:
        result["explosion_hypotheses"] = dbg.explain_failure("exploding_gradients")
    except Exception:
        result["explosion_hypotheses"] = []
    try:
        result["opt_hypotheses"] = dbg.explain_failure("optimizer_instability")
    except Exception:
        result["opt_hypotheses"] = []
    try:
        result["data_hypotheses"] = dbg.explain_failure("data_anomaly")
    except Exception:
        result["data_hypotheses"] = []
    if rl_detector is not None:
        result["rl_summary"] = rl_detector.summary()
        result["rl_events"] = rl_detector.dump_events()
    return result


def scenario_policy_collapse(num_steps=30):
    """Policy with extreme init -> all actions collapse to one, vanishing gradients."""
    model = ActorCritic(state_dim=8, action_dim=4, hidden=32)
    with torch.no_grad():
        for name, param in model.policy.named_parameters():
            if "weight" in name:
                param.fill_(0.0)
    return train_ppo(model, num_steps=num_steps, lr=1e-5)


def scenario_value_explosion(num_steps=30):
    """Value network with inflated weights -> exploding value gradients."""
    model = ActorCritic(state_dim=8, action_dim=4, hidden=32)
    with torch.no_grad():
        for name, param in model.value.named_parameters():
            if "weight" in name:
                param.mul_(1000.0)
    return train_ppo(model, num_steps=num_steps, lr=1e-1)


def scenario_reward_hacking(num_steps=30):
    """Extreme reward scale -> policy learns to exploit, gradient instability."""
    model = ActorCritic(state_dim=8, action_dim=4, hidden=32)
    return train_ppo(model, num_steps=num_steps, lr=3e-4, reward_scale=1e6)


def scenario_logit_saturation(num_steps=30):
    """Logits pushed to extreme values → softmax becomes one-hot, policy can't learn."""
    model = ActorCritic(state_dim=8, action_dim=4, hidden=32)
    with torch.no_grad():
        for name, param in model.policy.named_parameters():
            if "weight" in name:
                param.mul_(50.0)
    return train_ppo(model, num_steps=num_steps, lr=1e-3)


def scenario_reward_variance_collapse(num_steps=30):
    """All rewards identical → no learning signal, value can't fit."""
    model = ActorCritic(state_dim=8, action_dim=4, hidden=32)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    rl_detector = RLDetector(model)

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0,
                   family="RL") as dbg:
        for step in range(num_steps):
            optimizer.zero_grad()
            dbg.step = step

            states, actions, _, _ = _make_dummy_batch()
            # All rewards = 0.5 (no variance)
            rewards = torch.full_like(torch.randn(16), 0.5)
            logits, values = model(states)

            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            advantages = _compute_advantages(rewards, values)
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = ((values - rewards) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            loss.backward()
            dbg.record_loss(loss.item())
            rl_detector.step(logits, actions, rewards, values, step=step)
            optimizer.step()

    # Merge RL detector events into NeuralDBG events
    for rl_event in rl_detector.dump_events():
        merged = type('RLEvent', (), {
            'event_type': rl_event['event_type'],
            'step': rl_event['step'],
            'detail': rl_event['detail'],
            'severity': rl_event['severity'],
            'metadata': rl_event['metadata'],
            'layer_name': 'rl_policy',
            'from_state': 'healthy',
            'to_state': rl_event['event_type'],
            'to_dict': lambda self, d=rl_event: d,
        })()
        dbg.events.append(merged)

    return dbg, rl_detector


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] RL (PPO-style) failure scenarios + RLDetector\n")
    print("RLDetector solves 4 blind spots:")
    print("  1. Raw logit hooks (before log_softmax compression)")
    print("  2. Reward variance + spike tracking")
    print("  3. Policy collapse (entropy + action diversity)")
    print("  4. Adaptive thresholds (reward-scale aware)\n")

    total_detected = 0
    total_scenarios = 0

    for name, fn in [
        ("Policy collapse -> vanishing", scenario_policy_collapse),
        ("Value explosion -> exploding", scenario_value_explosion),
        ("Reward hacking -> instability", scenario_reward_hacking),
        ("Logit saturation -> dead softmax", scenario_logit_saturation),
        ("Reward variance collapse -> no signal", scenario_reward_variance_collapse),
    ]:
        total_scenarios += 1
        dbg, rl_detector = fn(num_steps=20)
        results = analyze_results(dbg, rl_detector)
        rl_events = results.get("rl_events", [])
        rl_summary = results.get("rl_summary", {})

        detected = len(rl_events) > 0
        if detected:
            total_detected += 1

        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {name}")
        print(f"  NeuralDBG events: {len(results['events'])}")
        print(f"  RLDetector events: {len(rl_events)} {'✅ DETECTED' if detected else '❌ MISSED'}")
        if rl_summary:
            print(f"  RL event types: {rl_summary.get('events_by_type', {})}")
            print(f"  Reward stats: {rl_summary.get('reward_stats', {})}")
            print(f"  Entropy stats: {rl_summary.get('entropy_stats', {})}")
        for e in rl_events[:3]:
            print(f"    [{e['event_type']}] step={e['step']}: {e['detail'][:80]}")

    print(f"\n{'=' * 60}")
    print(f"RL DETECTION: {total_detected}/{total_scenarios} ({total_detected/total_scenarios*100:.0f}%)")
    print("(was 0/36 = 0% across all RL configs before RLDetector)")


if __name__ == "__main__":
    main()


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] RL (PPO-style) failure scenarios\n")

    for name, fn in [
        ("Policy collapse -> vanishing", scenario_policy_collapse),
        ("Value explosion -> exploding", scenario_value_explosion),
        ("Reward hacking -> instability", scenario_reward_hacking),
    ]:
        dbg = fn(num_steps=20)
        results = analyze_results(dbg)
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {name}")
        print(f"{'=' * 60}")
        print(f"Events: {len(results['events'])}")
        for label, hyps in [
            ("Gradient hypotheses", results["hypotheses"]),
            ("Optimizer hypotheses", results["opt_hypotheses"]),
            ("Data anomaly", results["data_hypotheses"]),
        ]:
            if hyps:
                print(f"{label}:")
                for h in hyps:
                    print(f"  [{h.confidence:.2f}] {h.description}")
        if results["couplings"]:
            print("Coupled failures:")
            for c in results["couplings"]:
                d = c.get("step_difference", 0)
                print(
                    f"  {c['trigger']} -> {c['consequence']} (d={d}, {c['confidence']:.2f})"
                )

    print("\n[DONE] RL scenarios complete.")


if __name__ == "__main__":
    main()
