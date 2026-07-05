"""One-click NeuralDBG demo — catch a random bug in 60 seconds.

Trains a small model, injects a random failure, NeuralDBG detects it,
shows the causal chain, and exports JSON for Aquarium visualization.

Usage: python demo_one_click.py
"""

import sys, json, random, webbrowser
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")

import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg

torch.manual_seed(random.randint(0, 1000))

# ============================================================
# Model + Data
# ============================================================
class DemoModel(nn.Module):
    """Small but realistic model with mixed layer types."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16, 32)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.lin1 = nn.Linear(32, 64)
        self.lin2 = nn.Linear(64, 32)
        self.head = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.embed(x))
        x = x.unsqueeze(1).repeat(1, 4, 1)  # [B,16] -> [B,4,16]
        x, _ = self.lstm(x)
        x = F.relu(self.lin1(x[:, -1, :]))
        x = F.relu(self.lin2(x))
        return self.head(x)


def make_data(batch=16):
    return torch.randn(batch, 16), torch.randint(0, 2, (batch,))


# ============================================================
# Bug definitions
# ============================================================
BUGS = {
    "Exploding LR": lambda opt: setattr(opt.param_groups[0], 'lr', 50.0),
    "Vanishing (LSTM forget gate)": lambda opt, model: corrupt_forget_gate(model),
    "NaN in data": lambda x: x.index_put_([torch.tensor([0]), torch.tensor([0])], torch.tensor(float('nan'))),
    "Zero init": lambda model: [nn.init.zeros_(p) for p in model.parameters() if p.dim() >= 2],
    "Dead bias": lambda model: [nn.init.constant_(m.bias, -10.0) for m in model.modules()
                                if hasattr(m, 'bias') and m.bias is not None],
}

def corrupt_forget_gate(model):
    for m in model.modules():
        if isinstance(m, nn.LSTM):
            with torch.no_grad():
                hs = m.hidden_size
                if hasattr(m, 'bias_hh_l0'):
                    m.bias_hh_l0[hs:2*hs] = -10.0


# ============================================================
# Run
# ============================================================
print("=" * 55)
print("  NEURALDBG — One-Click Bug Hunt")
print("=" * 55)

bug_name = random.choice(list(BUGS.keys()))
print(f"  Bug: {bug_name}")
print()

model = DemoModel()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
bug_fn = BUGS[bug_name]

with NeuralDbg(model) as dbg:
    for step in range(12):
        x, y = make_data()

        if step == 3:  # Inject bug at step 4
            if "LR" in bug_name:
                bug_fn(opt)
            elif "NaN" in bug_name:
                x = bug_fn(x)
            elif "init" in bug_name.lower() or "bias" in bug_name.lower():
                bug_fn(model)
            elif "forget" in bug_name.lower():
                bug_fn(opt, model)

        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        opt.step()

    events = dbg.dump_events()
    chains = dbg.explain_causal()
    hyps = dbg.explain_failure()

# Results
print("  Detection:")
n = len(events)
print(f"    Events captured: {n}")

if chains:
    c = chains[0]
    print(f"    Causal chain: {c.root_cause} -> {c.final_symptom}")
    print(f"    Confidence: {c.confidence:.0%} | Links: {len(c.links)}")
    print(f"    Evidence: {c.links[0].evidence[:80]}...")

if hyps:
    print(f"    Top hypothesis: {hyps[0].description[:100]}")

# Export Aquarium JSON
export = {
    "events": [{
        "event_type": e.event_type.value if hasattr(e, 'event_type') else str(e),
        "layer_name": e.layer_name if hasattr(e, 'layer_name') else "?",
        "step": e.step if hasattr(e, 'step') else 0,
        "from_state": e.from_state if hasattr(e, 'from_state') else "?",
        "to_state": e.to_state if hasattr(e, 'to_state') else "?",
        "confidence": e.confidence if hasattr(e, 'confidence') else 1.0,
    } for e in events],
    "causal_chains": [{"root_cause": c.root_cause, "final_symptom": c.final_symptom,
                       "confidence": c.confidence, "nodes": len(c.links)} for c in chains],
    "meta": {"bug": bug_name, "model": "DemoModel (Linear+LSTM+MLP)", "steps": 12}
}

json_path = "demo_export.json"
with open(json_path, "w") as f:
    json.dump(export, f, indent=2, default=str)

print(f"\n  Exported: {json_path}")
print(f"  Open: docs/aquarium.html and drag {json_path} into it")

print(f"\n{'='*55}")
print(f"  Bug: {bug_name}")
detected = n > 5
print(f"  Detected: {'YES' if detected else 'MISS'}")
print(f"  Chain: {chains[0].root_cause} -> {chains[0].final_symptom}" if chains else "  No chain")
print(f"{'='*55}")
print("  Run again: python demo_one_click.py")
