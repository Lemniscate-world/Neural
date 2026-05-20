#!/usr/bin/env python3
"""
torch.compile (Dynamo) compatibility scenarios demonstrating NeuralDBG causal inference.
Covers: hook integrity under compilation, gradient capture with compiled models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg


class SimpleMLP(nn.Module):
    def __init__(self, input_size=32, hidden=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, input_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


def _make_loader(input_size=32, num_samples=100, batch_size=8):
    X = torch.randn(num_samples, input_size)
    y = torch.randn(num_samples, input_size)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def train_compiled(model, dataloader, num_steps=20, lr=1e-3, use_compile=True):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if use_compile:
        model = torch.compile(model)

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                dbg.step = step
                output = model(batch_x)
                loss = criterion(output, batch_y)
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


def scenario_compile_healthy(num_steps=30):
    """torch.compile with healthy training -> hooks should still work."""
    model = SimpleMLP(input_size=32, hidden=64, num_layers=3)
    loader = _make_loader(num_samples=50)
    return train_compiled(model, loader, num_steps=num_steps, lr=1e-3, use_compile=True)


def scenario_compile_vanishing(num_steps=30):
    """torch.compile with tiny init -> vanishing gradients under compilation."""
    model = SimpleMLP(input_size=32, hidden=64, num_layers=3)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1e-8)
    loader = _make_loader(num_samples=50)
    return train_compiled(model, loader, num_steps=num_steps, lr=1e-6, use_compile=True)


def scenario_compile_exploding(num_steps=30):
    """torch.compile with inflated weights -> exploding gradients under compilation."""
    model = SimpleMLP(input_size=32, hidden=64, num_layers=3)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1000.0)
    loader = _make_loader(num_samples=50)
    return train_compiled(model, loader, num_steps=num_steps, lr=1e-1, use_compile=True)


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] torch.compile (Dynamo) failure scenarios\n")

    for name, fn in [
        ("Compiled healthy training", scenario_compile_healthy),
        ("Compiled + tiny init -> vanishing", scenario_compile_vanishing),
        ("Compiled + inflated -> exploding", scenario_compile_exploding),
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

    print("\n[DONE] torch.compile scenarios complete.")


if __name__ == "__main__":
    main()
