#!/usr/bin/env python3
"""
LSTM / Time-Series failure scenarios demonstrating NeuralDBG causal inference.
Covers: vanishing recurrent gradients, exploding recurrent gradients, dead cells.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, use_tanh=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_tanh = use_tanh
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def _make_ts_loader(seq_len=20, input_size=4, num_samples=100, batch_size=8):
    X = torch.randn(num_samples, seq_len, input_size)
    y = torch.randn(num_samples, input_size)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def train_lstm(model, dataloader, num_steps=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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


def _zero_recurrent_weights(model):
    for name, param in model.named_parameters():
        if "weight_hh" in name:
            with torch.no_grad():
                param.fill_(0.0)


def _scale_recurrent_weights(model, factor):
    for name, param in model.named_parameters():
        if "weight_hh" in name:
            with torch.no_grad():
                param.mul_(factor)


def scenario_vanishing_recurrent(num_steps=30):
    """LSTM with zeroed recurrent weights -> vanishing recurrent gradients."""
    model = LSTMForecaster(input_size=4, hidden_size=32, num_layers=3)
    _zero_recurrent_weights(model)
    loader = _make_ts_loader(num_samples=50)
    return train_lstm(model, loader, num_steps=num_steps, lr=1e-3)


def scenario_exploding_recurrent(num_steps=30):
    """LSTM with inflated recurrent weights -> exploding recurrent gradients."""
    model = LSTMForecaster(input_size=4, hidden_size=32, num_layers=3)
    _scale_recurrent_weights(model, factor=50.0)
    loader = _make_ts_loader(num_samples=50)
    return train_lstm(model, loader, num_steps=num_steps, lr=1e-2)


def scenario_deep_lstm(num_steps=30):
    """Deep LSTM (6 layers) -> vanishing gradients in early layers."""
    model = LSTMForecaster(input_size=4, hidden_size=16, num_layers=6)
    loader = _make_ts_loader(num_samples=50)
    return train_lstm(model, loader, num_steps=num_steps, lr=1e-4)


def main():
    import os

    torch.manual_seed(42)
    print("[NeuralDBG] LSTM / Time-Series failure scenarios\n")

    os.makedirs("aquarium_exports", exist_ok=True)

    for name, fn in [
        ("Zeroed recurrent weights -> vanishing", scenario_vanishing_recurrent),
        ("Inflated recurrent weights -> exploding", scenario_exploding_recurrent),
        ("Deep LSTM (6 layers) -> vanishing early", scenario_deep_lstm),
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

        safe_name = name.replace(" ", "_").replace("->", "to").lower()[:40]
        export_path = f"aquarium_exports/{safe_name}.json"
        dbg.export_aquarium_package(export_path)
        print(f"  Exported: {export_path}")

    print("\n[DONE] LSTM scenarios complete.")


if __name__ == "__main__":
    main()
