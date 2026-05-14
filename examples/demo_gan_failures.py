#!/usr/bin/env python3
"""
GAN (generator-only) failure scenarios demonstrating NeuralDBG causal inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg


class Generator(nn.Module):
    def __init__(self, noise_dim=32, hidden_dim=64, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def _make_loader(noise_dim=32, num_samples=100, batch_size=8):
    z = torch.randn(num_samples, noise_dim)
    y = torch.randint(0, 2, (num_samples,))
    return DataLoader(TensorDataset(z, y), batch_size=batch_size, shuffle=True)


def train_generator(model, dataloader, num_steps=20, lr=0.001, nan_step=None):
    """Train generator with NeuralDBG monitoring."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            for batch_z, batch_y in dataloader:
                optimizer.zero_grad()
                dbg.step = step

                if (
                    nan_step is not None
                    and step >= nan_step
                    and not (hasattr(model, "_nan_injected") and model._nan_injected)
                ):
                    model.net[0].weight.data[0, 0] = float("nan")
                    model._nan_injected = True

                output = model(batch_z)
                loss = criterion(output[:, 0], batch_y.float())
                loss.backward()
                dbg.record_loss(loss.item())
                optimizer.step()
                break
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


def scenario_vanishing_generator(num_steps=20):
    """Generator with extremely low LR -> gradients vanish."""
    model = Generator()
    loader = _make_loader()
    return train_generator(model, loader, num_steps=num_steps, lr=1e-8)


def scenario_exploding_generator(num_steps=20):
    """Generator with extremely high LR -> gradients explode."""
    model = Generator()
    loader = _make_loader()
    return train_generator(model, loader, num_steps=num_steps, lr=1e4)


def scenario_generator_nan(num_steps=20):
    """Generator with NaN injection -> data anomaly."""
    model = Generator()
    loader = _make_loader()
    return train_generator(model, loader, num_steps=num_steps, lr=0.01, nan_step=5)


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] GAN Generator failure scenarios\n")

    for name, fn in [
        ("Vanishing gradients (LR=1e-8)", scenario_vanishing_generator),
        ("Exploding gradients (LR=1e4)", scenario_exploding_generator),
        ("NaN injection at step 5", scenario_generator_nan),
    ]:
        dbg = fn(num_steps=15)
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
            for c in results["couplings"][:5]:
                d = c.get("step_difference", 0)
                print(
                    f"  {c['trigger']} -> {c['consequence']} (d={d}, {c['confidence']:.2f})"
                )

    print("\n[DONE] GAN generator scenarios complete.")


if __name__ == "__main__":
    main()
