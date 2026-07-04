"""Capture live NeuralDBG events for BUG-001,003,005,008,010.
Saves to neuralagent/data/triplets/live_events.jsonl (appends).
Run: python capture_live_events.py
"""
import sys, json, os
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")
import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg

torch.manual_seed(42)
OUT = r"C:\Users\Utilisateur\Documents\Neural-Agent\neuralagent\data\triplets\live_events.jsonl"
captured = []

def save_triplet(bug_id, category, events, hypotheses, chains):
    event_summary = []
    for e in events[-10:]:  # last 10 events
        event_summary.append({
            "type": e.get("event_type","?"),
            "layer": e.get("layer_name","?"),
            "step": e.get("step",0),
            "state": e.get("to_state", e.get("from_state","?")),
        })
    hyp_text = "\n".join(f"- [{h.confidence:.2f}] {h.description[:120]}" for h in hypotheses[:5])
    chain_text = ""
    if chains:
        c = chains[0]
        chain_text = f"Root: {c.root_cause} -> {c.final_symptom}. Path: {c.description[:200]}"

    instruction = f"Analyze these NeuralDBG events. Bug: {bug_id}, category: {category}. Events detected: {len(events)}. Hypotheses:\n{hyp_text}\nCausal chain: {chain_text}"
    response = json.dumps({
        "category": category,
        "bug_id": bug_id,
        "events_detected": len(events),
        "event_types": list(set(e.get("event_type","?") for e in events)),
        "top_hypothesis": hypotheses[0].description[:150] if hypotheses else "none",
        "causal_root": chains[0].root_cause if chains else "no chain",
        "causal_symptom": chains[0].final_symptom if chains else "no chain",
    })
    captured.append({"instruction": instruction, "response": response})
    print(f"  {bug_id}: {len(events)} events, {len(hypotheses)} hypotheses, {len(chains)} chains")

# ============================================================
# BUG-001: MHA fully masked row -> NaN
# ============================================================
print("BUG-001: MHA NaN gradients...")
class MHA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(16, 4, batch_first=True)
        self.lin = nn.Linear(16, 2)
    def forward(self, x):
        o, _ = self.mha(x, x, x)
        return self.lin(o.mean(1))

model = MHA_Model()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
target = torch.randn(4, 2)
x_bug = torch.randn(4, 8, 16)
# Fully mask row 0 to trigger NaN
attn_mask = torch.zeros(8, 8)
attn_mask[0, :] = float('-inf')

with NeuralDbg(model) as dbg:
    for s in range(8):
        opt.zero_grad()
        out = model(x_bug)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        opt.step()
    events = dbg.dump_events()
    hyps = dbg.explain_failure()
    chains = dbg.explain_causal()
save_triplet("BUG-001", "mha_fully_masked_row", events, hyps, chains)

# ============================================================
# BUG-003: Gradient explosion (huge inputs)
# ============================================================
print("BUG-003: Gradient explosion...")
class ExplodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x): return self.net(x)

model = ExplodeModel()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
target = torch.randn(4, 2)

with NeuralDbg(model) as dbg:
    for s in range(5):
        opt.zero_grad()
        x = torch.randn(4, 8) * (100.0 if s >= 2 else 1.0)  # explosion at step 2
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        opt.step()
    events = dbg.dump_events()
    hyps = dbg.explain_failure()
    chains = dbg.explain_causal()
save_triplet("BUG-003", "gradient_explosion", events, hyps, chains)

# ============================================================
# BUG-005: LSTM batch pollution (NaN in one sample)
# ============================================================
print("BUG-005: LSTM batch pollution...")
class LSTMModel(nn.Module):
    def __init__(self): super().__init__()
    self.lstm = nn.LSTM(4, 8, batch_first=True)
    self.lin = nn.Linear(8, 2)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.lin(o[:, -1])

model = LSTMModel()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
target = torch.randn(4, 2)

with NeuralDbg(model) as dbg:
    for s in range(5):
        opt.zero_grad()
        x = torch.randn(4, 5, 4)
        if s >= 2:
            x[0] = float('nan')  # sample 0 corrupted
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        opt.step()
    events = dbg.dump_events()
    hyps = dbg.explain_failure()
    chains = dbg.explain_causal()
save_triplet("BUG-005", "lstm_sample_independence", events, hyps, chains)

# ============================================================
# BUG-008: F.normalize zero-vector gradient corruption
# ============================================================
print("BUG-008: F.normalize gradient corruption...")
class NormModel(nn.Module):
    def __init__(self): super().__init__()
    self.lin = nn.Linear(8, 2)
    def forward(self, x): return self.lin(F.normalize(x, dim=1))

model = NormModel()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
target = torch.randn(4, 2)

with NeuralDbg(model) as dbg:
    for s in range(5):
        opt.zero_grad()
        x = torch.zeros(4, 8) if s >= 2 else torch.randn(4, 8)
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        opt.step()
    events = dbg.dump_events()
    hyps = dbg.explain_failure()
    chains = dbg.explain_causal()
save_triplet("BUG-008", "data_anomaly", events, hyps, chains)

# ============================================================
# BUG-010: Inductor quantile tied values -> gradient mismatch
# ============================================================
print("BUG-010: Quantile tied values...")
class QuantileModel(nn.Module):
    def __init__(self): super().__init__()
    self.lin = nn.Linear(3, 2)
    def forward(self, x):
        q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]), dim=1)
        return self.lin(q.permute(1, 0))

model = QuantileModel()
opt = torch.optim.SGD(model.parameters(), lr=0.001)
target = torch.randn(3, 2)

with NeuralDbg(model) as dbg:
    for s in range(5):
        opt.zero_grad()
        x = torch.ones(3, 5) * 3.0 if s >= 2 else torch.randn(3, 5)
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        opt.step()
    events = dbg.dump_events()
    hyps = dbg.explain_failure()
    chains = dbg.explain_causal()
save_triplet("BUG-010", "gradient_explosion", events, hyps, chains)

# ============================================================
# Save
# ============================================================
with open(OUT, 'a', encoding='utf-8') as f:
    for c in captured:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')

print(f"\nSaved {len(captured)} live triplets to {OUT}")
print(f"Total live events in file: {sum(1 for _ in open(OUT))}")
