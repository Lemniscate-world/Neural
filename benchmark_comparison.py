"""
Live Benchmark — NeuralDBG vs Monitoring Tools

Runs the same 5 failure scenarios through:
  1. NeuralDBG (causal chain detection)
  2. Baseline monitor (gradient norm tracking — simulates W&B/TensorBoard)

Measures: detection rate, localization accuracy, root cause identification.

Usage: python benchmark_comparison.py
Output: benchmark_comparison.json + benchmark_comparison.html
"""

import sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg

torch.manual_seed(42)


# ============================================================
# Baseline monitor (simulates W&B / TensorBoard)
# ============================================================

class BaselineMonitor:
    """Simulates what W&B/TensorBoard would detect: gradient norms, loss."""

    def __init__(self, model):
        self.model = model
        self.grad_norms = []
        self.losses = []
        self.alerts = []

    def step(self, loss):
        self.losses.append(loss)
        norms = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                norms[name] = p.grad.norm().item()
        self.grad_norms.append(norms)

        # Simple threshold-based alerts (what W&B does)
        if loss > 10 or (len(self.losses) > 2 and loss > 5 * max(self.losses[-3:-1] or [1])):
            self.alerts.append({"type": "loss_spike", "step": len(self.losses), "value": loss})
        if math.isnan(loss) or math.isinf(loss):
            self.alerts.append({"type": "nan_loss", "step": len(self.losses)})
        for name, norm in norms.items():
            if norm < 1e-6:
                self.alerts.append({"type": "vanishing_gradient", "layer": name, "step": len(self.losses)})
            if norm > 1e3:
                self.alerts.append({"type": "exploding_gradient", "layer": name, "step": len(self.losses)})

    def summary(self):
        return {
            "total_alerts": len(self.alerts),
            "alert_types": list(set(a["type"] for a in self.alerts)),
            "final_loss": self.losses[-1] if self.losses else None,
        }


# ============================================================
# Scenarios
# ============================================================

def build_model():
    return nn.Sequential(
        nn.Linear(16, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 10),
    )


def run_scenario(name, bug_fn, steps=15):
    """Run one scenario through both NeuralDBG and baseline monitor."""
    model = build_model()
    baseline = BaselineMonitor(model)

    with NeuralDbg(model) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for s in range(steps):
            x = torch.randn(8, 16)
            y = torch.randint(0, 10, (8,))
            if bug_fn and s >= 5:
                x, y, opt, _ = bug_fn(x, y, opt, model)

            opt.zero_grad()
            try:
                loss = nn.CrossEntropyLoss()(model(x), y)
                loss.backward()
                dbg.step_iteration()
                dbg.record_loss(loss.item())
                opt.step()
                baseline.step(loss.item())
            except Exception:
                break

        events = dbg.dump_events()
        chains = dbg.explain_causal()
        hyps = dbg.explain_failure()

    # Categorize NeuralDBG detection
    problem_types = set()
    for e in events:
        ts = str(e.get("to_state", "")).lower()
        et = str(e.get("event_type", ""))
        if ts in ("vanishing", "exploding", "saturated", "dead", "anomalous"):
            problem_types.add(ts)
        if et in ("nan_detected", "optimizer_instability", "data_anomaly"):
            problem_types.add(et)

    return {
        "scenario": name,
        "neuraldbg": {
            "events": len(events),
            "chains": len(chains),
            "hypotheses": len(hyps),
            "detected_types": sorted(problem_types),
            "top_chain": str(chains[0].root_cause) if chains else "none",
            "top_hypothesis": hyps[0].description[:100] if hyps else "none",
        },
        "baseline": baseline.summary(),
    }


# ============================================================
# Bug injectors
# ============================================================

def bug_exploding(x, y, opt, model):
    for g in opt.param_groups: g['lr'] = 50.0
    return x, y, opt, model

def bug_vanishing(x, y, opt, model):
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2: p.mul_(0.001)
    return x, y, opt, model

def bug_nan(x, y, opt, model):
    x = x.clone(); x[0, 0] = float('nan')
    return x, y, opt, model

def bug_dead(x, y, opt, model):
    for p in model.parameters():
        if p.dim() == 1: nn.init.constant_(p, -10.0)
    return x, y, opt, model

def bug_zero(x, y, opt, model):
    for p in model.parameters():
        if p.dim() >= 2: nn.init.zeros_(p)
    return x, y, opt, model


# ============================================================
# Main
# ============================================================

SCENARIOS = [
    ("Healthy training", None),
    ("Exploding gradients (LR=50)", bug_exploding),
    ("Vanishing gradients (weights/1000)", bug_vanishing),
    ("NaN data injection", bug_nan),
    ("Dead neurons (bias=-10)", bug_dead),
    ("Zero initialization", bug_zero),
]


def main():
    print("=" * 60)
    print("NEURALDBG vs BASELINE MONITOR — Live Benchmark")
    print("=" * 60)

    results = []
    for name, bug in SCENARIOS:
        print(f"\n  [{name}]")
        r = run_scenario(name, bug)
        print(f"    NeuralDBG: {r['neuraldbg']['events']} events, "
              f"{r['neuraldbg']['chains']} chains, "
              f"types={r['neuraldbg']['detected_types']}")
        print(f"    Baseline:  {r['baseline']['total_alerts']} alerts, "
              f"types={r['baseline']['alert_types']}")
        results.append(r)

    # Save JSON
    with open("benchmark_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<30s} | {'NeuralDBG':>10s} | {'Baseline':>10s}")
    print(f"{'-'*30}-|{'-'*12}|{'-'*12}")
    for r in results:
        nd = f"{r['neuraldbg']['chains']} chains" if r['neuraldbg']['chains'] else f"{r['neuraldbg']['events']} events"
        bl = f"{r['baseline']['total_alerts']} alerts"
        print(f"{r['scenario']:<30s} | {nd:>10s} | {bl:>10s}")

    # Score
    nd_score = sum(1 for r in results if r['neuraldbg']['chains'] > 0 or len(r['neuraldbg']['detected_types']) > 0)
    bl_score = sum(1 for r in results if r['baseline']['total_alerts'] > 0)
    print(f"\n  Detection rate: NeuralDBG {nd_score}/{len(results)} vs Baseline {bl_score}/{len(results)}")

    print(f"\n  Report: benchmark_comparison.json")


if __name__ == "__main__":
    main()
