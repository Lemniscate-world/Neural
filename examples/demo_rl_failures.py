#!/usr/bin/env python3
"""
RL (PPO-style / DQN-style) failure scenarios demonstrating NeuralDBG causal inference.
Covers: reward hacking, policy collapse, value explosion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from neuraldbg import NeuralDbg


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


def train_ppo(model, num_steps=20, lr=3e-4, state_dim=8, action_dim=4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            optimizer.zero_grad()
            dbg.step = step

            states, actions, rewards, dones = _make_dummy_batch(state_dim, action_dim)
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
            optimizer.step()
    return dbg


def analyze_results(dbg):
    return {
        "hypotheses": dbg.explain_failure("vanishing_gradients")
        + dbg.explain_failure("exploding_gradients"),
        "opt_hypotheses": dbg.explain_failure("optimizer_instability"),
        "data_hypotheses": dbg.explain_failure("data_anomaly"),
        "couplings": dbg.detect_coupled_failures(),
        "events": dbg.events,
        "mermaid": dbg.export_mermaid_causal_graph(),
    }


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
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    gamma = 0.99

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            optimizer.zero_grad()
            dbg.step = step

            states, actions, rewards, dones = _make_dummy_batch()
            rewards = rewards * 1e6
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
            optimizer.step()
    return dbg


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
