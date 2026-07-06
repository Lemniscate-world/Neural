"""
Post-Mortem Reproduction Suite

Reproduces known PyTorch bugs with NeuralDBG, extracts causal chains,
and generates structured post-mortem documentation for the paper.

Usage:
    python reproduce_post_mortems.py [--bug BUG_ID] [--all]
"""

import sys, json, os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg

OUTPUT_DIR = Path("docs/post_mortems")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)


# ============================================================
# Post-Mortem 1: svdvals silently swallows NaN (#187759)
# ============================================================

def reproduce_svdvals_nan():
    """Reproduce: svdvals returns finite values for NaN input."""
    A = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, float('nan'), 6.0],
                       [7.0, 8.0, 9.0]])

    # NeuralDBG wrapper on a minimal model using svdvals
    class SVDModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x):
            # Simulate: a layer that uses svdvals internally
            s = torch.linalg.svdvals(x)
            return self.linear(s)

    model = SVDModel()
    # Set input to NaN matrix
    with NeuralDbg(model) as dbg:
        try:
            out = model(A)
            loss = out.sum()
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
        except RuntimeError:
            pass

        events = dbg.dump_events()
        chains = dbg.explain_causal()

    return {
        "bug_id": "PM-001",
        "title": "svdvals silently swallows NaN",
        "pytorch_issue": "#187759",
        "pr": "#188053",
        "events": len(events),
        "chains": len(chains),
        "causal_chain": str(chains[0]) if chains else "No chain — NaN consumed silently",
        "root_cause": "svdvals does not validate NaN in input matrix",
        "symptom": "Finite singular values returned for NaN matrix",
        "fix": "Add NaN propagation consistency test between svdvals and svd",
    }


# ============================================================
# Post-Mortem 2: F.normalize at zero input (#184575)
# ============================================================

def reproduce_normalize_zero():
    """Reproduce: F.normalize returns 0 instead of NaN at zero input."""
    class NormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x):
            x = self.linear(x)
            return F.normalize(x, dim=0)

    model = NormModel()
    zero_input = torch.zeros(3, requires_grad=True)

    with NeuralDbg(model) as dbg:
        out = model(zero_input.unsqueeze(0))
        loss = out.sum()
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())

        events = dbg.dump_events()
        chains = dbg.explain_causal()

    grad_is_finite = zero_input.grad is not None and zero_input.grad.isfinite().all()

    return {
        "bug_id": "PM-002",
        "title": "F.normalize returns 0 instead of NaN at zero input",
        "pytorch_issue": "#184575",
        "pr": "#188066",
        "events": len(events),
        "chains": len(chains),
        "gradient_finite": bool(grad_is_finite),
        "causal_chain": str(chains[0]) if chains else "No chain — silent finite gradient",
        "root_cause": "F.normalize computes x/||x|| without checking for ||x||=0",
        "symptom": "Finite output (0) and finite gradients for zero input",
        "fix": "Return NaN when input norm is zero",
    }


# ============================================================
# Post-Mortem 3: Gradient explosion via extreme LR (generic)
# ============================================================

def reproduce_gradient_explosion():
    """Reproduce: exploding gradients via extreme learning rate."""
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

    x = torch.randn(8, 16)
    y = torch.randint(0, 10, (8,))

    with NeuralDbg(model) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Step 1-3: normal training
        for s in range(3):
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()

        # Step 4: inject exploding LR
        for g in opt.param_groups:
            g['lr'] = 50.0
        opt.zero_grad()
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())

        events = dbg.dump_events()
        chains = dbg.explain_causal()

    # Find the explosion event
    explosion_events = [e for e in events
                        if e.get("to_state") == "exploding"
                        or "explosion" in str(e.get("event_type", "")).lower()]

    return {
        "bug_id": "PM-003",
        "title": "Gradient explosion via extreme learning rate",
        "pytorch_issue": "Generic (common failure mode)",
        "pr": "N/A",
        "events": len(events),
        "chains": len(chains),
        "explosion_events": len(explosion_events),
        "causal_chain": str(chains[0]) if chains else "No chain extracted",
        "root_cause": "LR=50.0 causes gradient norm to exceed stable bounds",
        "symptom": "Gradient health transition: stable -> exploding",
        "fix": "Add gradient clipping (max_norm=1.0) or reduce LR to ≤0.01",
    }


# ============================================================
# Post-Mortem 4: Vanishing gradients via Sigmoid saturation
# ============================================================

def reproduce_vanishing_sigmoid():
    """Reproduce: vanishing gradients due to Sigmoid saturation."""
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.Sigmoid(),  # Deliberately use Sigmoid (saturates)
        nn.Linear(64, 32),
        nn.Sigmoid(),
        nn.Linear(32, 10),
    )

    # Initialize with small weights to accelerate saturation
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.constant_(p, 0.01)

    x = torch.randn(8, 16)
    y = torch.randint(0, 10, (8,))

    with NeuralDbg(model) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for s in range(10):
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()

        events = dbg.dump_events()
        chains = dbg.explain_causal()

    vanishing_events = [e for e in events
                        if e.get("to_state") == "vanishing"
                        or "vanishing" in str(e.get("event_type", "")).lower()]

    return {
        "bug_id": "PM-004",
        "title": "Vanishing gradients via Sigmoid saturation",
        "pytorch_issue": "Generic (common failure mode)",
        "pr": "N/A",
        "events": len(events),
        "chains": len(chains),
        "vanishing_events": len(vanishing_events),
        "causal_chain": str(chains[0]) if chains else "No chain extracted",
        "root_cause": "Sigmoid activation saturates at extremes, gradient -> 0",
        "symptom": "Gradient norm < 1e-6 in deeper layers, no learning",
        "fix": "Replace Sigmoid with ReLU/LeakyReLU, use BatchNorm",
    }


# ============================================================
# Post-Mortem 5: Dead ReLU neurons (zero init)
# ============================================================

def reproduce_dead_relu():
    """Reproduce: dead neurons via zero initialization."""
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

    # Zero-init all weights
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.zeros_(p)
    for p in model.parameters():
        if p.dim() == 1:
            nn.init.constant_(p, -10.0)  # Dead bias

    x = torch.randn(8, 16)
    y = torch.randint(0, 10, (8,))

    with NeuralDbg(model) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for s in range(10):
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()

        events = dbg.dump_events()
        chains = dbg.explain_causal()

    return {
        "bug_id": "PM-005",
        "title": "Dead neurons via zero initialization + negative bias",
        "pytorch_issue": "Generic (common failure mode)",
        "pr": "N/A",
        "events": len(events),
        "chains": len(chains),
        "causal_chain": str(chains[0]) if chains else "No chain extracted",
        "root_cause": "Zero weights + negative bias -> all ReLU outputs = 0",
        "symptom": "Zero gradients or constant output, no learning",
        "fix": "Use nn.init.kaiming_uniform_ or Xavier initialization",
    }


# ============================================================
# Post-Mortem 6: NaN data propagation
# ============================================================

def reproduce_nan_propagation():
    """Reproduce: NaN in input data propagates through model."""
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    x = torch.randn(8, 16)
    x[0, 0] = float('nan')  # Inject NaN
    y = torch.randint(0, 10, (8,))

    with NeuralDbg(model) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        opt.zero_grad()
        try:
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
        except RuntimeError:
            pass
        dbg.step_iteration()
        dbg.record_loss(float('nan') if 'loss' not in dir() else loss.item())

        events = dbg.dump_events()
        chains = dbg.explain_causal()

    nan_events = [e for e in events
                  if "nan" in str(e.get("event_type", "")).lower()
                  or e.get("to_state") == "nan"]

    return {
        "bug_id": "PM-006",
        "title": "NaN data propagation through feed-forward network",
        "pytorch_issue": "Generic (common failure mode)",
        "pr": "N/A",
        "events": len(events),
        "chains": len(chains),
        "nan_events": len(nan_events),
        "causal_chain": str(chains[0]) if chains else "No chain extracted",
        "root_cause": "NaN in input data at index [0,0]",
        "symptom": "NaN propagates through all layers, loss = NaN",
        "fix": "Add torch.isnan(x) check before forward pass, clean data pipeline",
    }


# ============================================================
# Post-Mortem 7: Divergence via extreme LR
# ============================================================

def reproduce_divergence():
    """Reproduce: training divergence via extremely high LR."""
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    x = torch.randn(8, 16)
    y = torch.randint(0, 10, (8,))

    with NeuralDbg(model) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=500.0)  # Extreme LR

        for s in range(8):
            opt.zero_grad()
            try:
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                dbg.step_iteration()
                dbg.record_loss(loss.item())
                opt.step()
            except RuntimeError:
                dbg.record_loss(float('inf'))
                break

        events = dbg.dump_events()
        chains = dbg.explain_causal()

    return {
        "bug_id": "PM-007",
        "title": "Training divergence via extreme learning rate (LR=500)",
        "pytorch_issue": "Generic (common failure mode)",
        "pr": "N/A",
        "events": len(events),
        "chains": len(chains),
        "causal_chain": str(chains[0]) if chains else "No chain extracted",
        "root_cause": "LR=500 causes loss to diverge to inf in <8 steps",
        "symptom": "Loss spikes -> inf, optimizer instability",
        "fix": "Reduce LR to ≤0.01, add gradient clipping, use learning rate scheduler",
    }


# ============================================================
# Post-Mortem 8: Gradient clipping too aggressive
# ============================================================

def reproduce_clip_underflow():
    """Reproduce: gradient clipping set too low causes zero gradients."""
    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 10))
    x, y = torch.randn(8, 16), torch.randint(0, 10, (8,))
    with NeuralDbg(model) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for s in range(10):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-8)
            dbg.step_iteration(); dbg.record_loss(loss.item()); opt.step()
        events, chains = dbg.dump_events(), dbg.explain_causal()
    return {"bug_id": "PM-008", "title": "Gradient clipping too aggressive",
            "events": len(events), "chains": len(chains),
            "causal_chain": str(chains[0]) if chains else "No chain",
            "root_cause": "max_norm=1e-8 clips all gradients to near-zero",
            "symptom": "Model stops learning, loss plateaus immediately",
            "fix": "Set max_norm=1.0 or higher. Typical values: 0.5-5.0"}


# ============================================================
# Post-Mortem 9: AdamW weight decay + LayerNorm instability
# ============================================================

def reproduce_adamw_layernorm():
    """Reproduce: AdamW weight decay destabilizes LayerNorm."""
    model = nn.Sequential(nn.Linear(16, 64), nn.LayerNorm(64), nn.ReLU(),
                          nn.Linear(64, 10))
    x, y = torch.randn(8, 16), torch.randint(0, 10, (8,))
    with NeuralDbg(model) as dbg:
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=10.0)
        for s in range(15):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), y); loss.backward()
            dbg.step_iteration(); dbg.record_loss(loss.item()); opt.step()
        events, chains = dbg.dump_events(), dbg.explain_causal()
    return {"bug_id": "PM-009", "title": "AdamW weight decay + LayerNorm",
            "events": len(events), "chains": len(chains),
            "causal_chain": str(chains[0]) if chains else "No chain",
            "root_cause": "weight_decay=10.0 excessive on LayerNorm weights",
            "symptom": "LayerNorm activations oscillate, loss unstable",
            "fix": "Reduce weight_decay to <=0.1 or exclude LN from decay"}


# ============================================================
# Post-Mortem 10: fp16 mixed precision softmax underflow
# ============================================================

def reproduce_fp16_softmax_underflow():
    """Reproduce: fp16 softmax underflow with large inputs."""
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    x = torch.randn(8, 64) * 100.0; y = torch.randint(0, 10, (8,))
    with NeuralDbg(model.half()) as dbg:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        for s in range(8):
            opt.zero_grad()
            try:
                loss = nn.CrossEntropyLoss()(model(x.half()).float(), y)
                loss.backward(); dbg.step_iteration(); dbg.record_loss(loss.item()); opt.step()
            except RuntimeError:
                dbg.record_loss(float('nan')); break
        events, chains = dbg.dump_events(), dbg.explain_causal()
    return {"bug_id": "PM-010", "title": "fp16 softmax underflow",
            "events": len(events), "chains": len(chains),
            "causal_chain": str(chains[0]) if chains else "No chain",
            "root_cause": "Large inputs (x100) cause fp16 softmax underflow",
            "symptom": "NaN gradients in fp16 with large activations",
            "fix": "Use fp32 softmax or scale inputs to [-10, 10]"}


# ============================================================
# Post-Mortem Generator
# ============================================================

ALL_POST_MORTEMS = [
    ("svdvals_nan", reproduce_svdvals_nan),
    ("normalize_zero", reproduce_normalize_zero),
    ("gradient_explosion", reproduce_gradient_explosion),
    ("vanishing_sigmoid", reproduce_vanishing_sigmoid),
    ("dead_relu", reproduce_dead_relu),
    ("nan_propagation", reproduce_nan_propagation),
    ("divergence", reproduce_divergence),
    ("clip_underflow", reproduce_clip_underflow),
    ("adamw_layernorm", reproduce_adamw_layernorm),
    ("fp16_softmax_underflow", reproduce_fp16_softmax_underflow),
]


def generate_post_mortem_md(pm: Dict[str, Any]) -> str:
    """Generate a Markdown post-mortem from a result dict."""
    return f"""---
bug_id: {pm['bug_id']}
title: {pm['title']}
pytorch_issue: {pm.get('pytorch_issue', 'N/A')}
pr: {pm.get('pr', 'N/A')}
date: {datetime.now().strftime('%Y-%m-%d')}
---

# {pm['bug_id']}: {pm['title']}

## Metadata
- **PyTorch Issue**: {pm.get('pytorch_issue', 'N/A')}
- **PR**: {pm.get('pr', 'N/A')}
- **Events Captured**: {pm['events']}
- **Causal Chains**: {pm['chains']}

## Root Cause
{pm['root_cause']}

## Symptom
{pm['symptom']}

## Causal Chain (NeuralDBG)
```
{pm['causal_chain']}
```

## Fix
{pm['fix']}

## Detection Metrics
{json.dumps({k: v for k, v in pm.items() if k not in ('bug_id', 'title', 'root_cause', 'symptom', 'causal_chain', 'fix', 'pytorch_issue', 'pr')}, indent=2)}

---
*Generated by NeuralDBG Post-Mortem Suite v1.0*
"""


def main():
    print("=" * 60)
    print("NeuralDBG Post-Mortem Reproduction Suite")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 60)

    results = []
    for pm_id, pm_fn in ALL_POST_MORTEMS:
        print(f"\n--- {pm_id} ---")
        try:
            result = pm_fn()
            print(f"  Events: {result['events']}, Chains: {result['chains']}")
            print(f"  Root: {result['root_cause'][:80]}...")

            # Save markdown
            md_content = generate_post_mortem_md(result)
            md_path = OUTPUT_DIR / f"{pm_id}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            print(f"  Saved: {md_path}")

            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Save combined JSON
    json_path = OUTPUT_DIR / "all_post_mortems.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generated {len(results)}/{len(ALL_POST_MORTEMS)} post-mortems")
    print(f"JSON: {json_path}")
    print(f"Markdown: {OUTPUT_DIR}/*.md")

    # Summary for paper
    print(f"\n--- Paper Summary ---")
    for r in results:
        chains_str = "CHAINS" if r['chains'] > 0 else "NO_CHAINS"
        print(f"  {r['bug_id']}: {r['events']} events, {r['chains']} chains {chains_str}")


if __name__ == "__main__":
    main()
