"""
Tier 4b Black-Swan — Federated Learning

Simulates Federated Averaging (FedAvg) with multiple client models.
Key failure modes: client drift, aggregation collapse, communication dropout.

Usage: python validate_blackswans_federated.py
"""

import sys, json, random, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg
from validate_combinatorial import ArchConfig, BUGS, n_problematic

torch.manual_seed(42)
random.seed(42)


# ============================================================
# Federated Learning Simulator
# ============================================================

class ClientModel(nn.Module):
    """Simple model for each FL client."""

    def __init__(self, dim=32, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def fedavg_aggregate(global_model, client_models):
    """Federated Averaging: average client weights into global model."""
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.stack([
            m.state_dict()[key].float() for m in client_models
        ]).mean(0)
    global_model.load_state_dict(global_dict)


def fl_configs(n=6):
    configs = []
    dims = [32, 64, 128]
    clients = [3, 5]
    idx = 0
    for d in dims:
        for c in clients:
            if idx >= n: return configs
            configs.append(ArchConfig(
                family="Federated", name=f"FL_d{d}_c{c}",
                depth=2, width=d, activation="relu", norm=None,
                skip=False, dropout=0.0,
                extra={"num_clients": c, "local_steps": 3}))
            idx += 1
    return configs


# ============================================================
# Training
# ============================================================

def train_federated(cfg: ArchConfig, steps=8, bug=None):
    """Simulate Federated Learning with NeuralDBG on the global model."""
    num_clients = cfg.extra.get("num_clients", 3)
    local_steps = cfg.extra.get("local_steps", 3)
    dim = cfg.width

    global_model = ClientModel(dim=dim)
    client_models = [ClientModel(dim=dim) for _ in range(num_clients)]

    # Sync clients to global
    for cm in client_models:
        cm.load_state_dict(global_model.state_dict())

    opt = torch.optim.SGD(global_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    with NeuralDbg(global_model) as dbg:
        bug_applied = False
        for s in range(steps):
            # Each client trains locally
            for cm in client_models:
                x = torch.randn(8, dim)
                y = torch.randint(0, 10, (8,))
                client_opt = torch.optim.SGD(cm.parameters(), lr=0.01)
                for _ in range(local_steps):
                    client_opt.zero_grad()
                    loss = loss_fn(cm(x), y)
                    loss.backward()
                    client_opt.step()

            # Aggregate: FedAvg
            fedavg_aggregate(global_model, client_models)

            # Apply bug to global model if needed
            if bug and s >= 3 and not bug_applied:
                x_dummy = torch.randn(8, dim)
                y_dummy = torch.randint(0, 10, (8,))
                _, _, opt, _ = bug(x_dummy, y_dummy, opt, global_model)
                bug_applied = True
                # Re-sync clients after bug
                for cm in client_models:
                    cm.load_state_dict(global_model.state_dict())

            # Evaluate on global model
            x_eval = torch.randn(8, dim)
            y_eval = torch.randint(0, 10, (8,))
            opt.zero_grad()
            try:
                loss = loss_fn(global_model(x_eval), y_eval)
                loss.backward()
                dbg.step_iteration()
                dbg.record_loss(loss.item())
                opt.step()
            except Exception:
                break

        events = dbg.dump_events()
        hyps = dbg.explain_failure()
        chains = dbg.explain_causal()
    return events, hyps, chains


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BLACK-SWAN ARCHITECTURE TESTER — Tier 4b: Federated Learning")
    print("=" * 60)

    all_configs = fl_configs(6)
    bugs_to_test = BUGS

    results = []
    for cfg in all_configs:
        print(f"\n  [Federated] {cfg.name}")

        try:
            ev, _, _ = train_federated(cfg, steps=5)
            baseline = n_problematic(ev)
        except Exception as e:
            print(f"    Baseline error: {e}")
            continue

        threshold = max(baseline + 2, 2)
        detected = 0
        total = 0

        for bug_name, bug_fn in bugs_to_test:
            try:
                ev, _, _ = train_federated(cfg, steps=5, bug=bug_fn)
                n = n_problematic(ev)
                if n > threshold:
                    detected += 1
                total += 1
            except Exception as e:
                total += 1

        print(f"    Baseline: {baseline} | Detected: {detected}/{total}")
        results.append({
            "family": "Federated", "name": cfg.name,
            "detected": detected, "total": total
        })

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS — Tier 4b: Federated Learning")
    print(f"{'='*60}")

    total_d, total_t = 0, 0
    for r in results:
        total_d += r["detected"]
        total_t += r["total"]

    pct = total_d / max(total_t, 1) * 100
    print(f"  Federated: {total_d}/{total_t} ({pct:.0f}%)")

    report = {
        "tier": "4b",
        "family": "Federated",
        "detected": total_d,
        "total": total_t,
        "pct": pct,
        "details": results,
    }
    with open("blackswan_federated_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: blackswan_federated_results.json")
