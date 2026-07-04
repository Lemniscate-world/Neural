"""End-to-end RNN pipeline: NeuralDBG -> Causal Chain -> Neural-Agent -> Fix -> Validate.

Proves the full NeuralSuite closed loop on recurrent architectures.
Runs entirely on CPU — uses rules-based agent (no GPU needed for demo).

Usage: python e2e_rnn_pipeline.py
"""

import sys, json
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\Neural-Agent")

import torch, torch.nn as nn
import torch.nn.functional as F
from neuraldbg import NeuralDbg
from neuralagent.remediator import Remediator

torch.manual_seed(42)

# ============================================================
# Mini RNN model
# ============================================================
class RNNDemo(nn.Module):
    def __init__(self, input_dim=16, hidden=32, num_layers=2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden * mult, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_rnn_data(batch=16, seq_len=16, dim=16):
    x = torch.randn(batch, seq_len, dim)
    y = torch.randint(0, 2, (batch,))
    return x, y


# ============================================================
# Training with NeuralDBG
# ============================================================
def train_with_dbg(model, data_fn, steps=12, lr=0.01, bug_fn=None, bug_step=4):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    with NeuralDbg(model) as dbg:
        for s in range(steps):
            x, y = data_fn()
            if bug_fn and s >= bug_step:
                x, y, opt = bug_fn(x, y, opt)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()

        events = dbg.dump_events()
        hyps = dbg.explain_failure()
        chains = dbg.explain_causal()

    return events, hyps, chains


# ============================================================
# Bug injectors
# ============================================================
def bug_exploding(x, y, opt):
    for g in opt.param_groups:
        g['lr'] = 50.0
    return x, y, opt

def bug_vanishing_rnn(x, y, opt):
    # Corrupt LSTM forget gate bias -> everything forgotten -> BPTT vanishing
    for m in model_ref[0].modules():
        if isinstance(m, nn.LSTM) and hasattr(m, 'bias_hh_l0'):
            with torch.no_grad():
                hs = m.hidden_size
                m.bias_hh_l0[hs:2*hs] = -10.0  # Forget gate
    return x, y, opt

def bug_nan_data(x, y, opt):
    x = x.clone()
    x[0, 0, 0] = float('nan')
    return x, y, opt

def bug_zero_init(x, y, opt):
    for p in model_ref[0].parameters():
        if p.dim() >= 2:
            nn.init.zeros_(p)
    return x, y, opt

model_ref = []


# ============================================================
# Pipeline runner
# ============================================================
def pipeline(bug_name, bug_fn, steps=12):
    global model_ref
    print(f"\n{'='*55}")
    print(f"  PIPELINE: {bug_name}")
    print(f"{'='*55}")

    # Phase 1: Detect
    print("  [1/4] DETECT — Training with NeuralDBG hooks...")
    model = RNNDemo(hidden=32, num_layers=2)
    model_ref = [model]
    data_fn = make_rnn_data

    events, hyps, chains = train_with_dbg(model, data_fn, steps=steps, bug_fn=bug_fn)

    n_anomalies = 0
    for e in events:
        et = e.event_type.value if hasattr(e, 'event_type') else e.get('event_type', '')
        ts = (e.to_state if hasattr(e, 'to_state') else e.get('to_state', '')).lower() if hasattr(e, 'to_state') or isinstance(e, dict) else ''
        if et == 'activation_regime_shift' and ts in ('normal', 'healthy', 'none', ''):
            continue
        n_anomalies += 1
    top_chain = f"{chains[0].root_cause} -> {chains[0].final_symptom}" if chains else "no chain"
    top_hyp = hyps[0].description if hyps else "no hypothesis"
    print(f"    Anomalies: {n_anomalies} | Chain: {top_chain[:60]}")

    # Phase 2: Diagnose
    print("  [2/4] DIAGNOSE — Running Neural-Agent rules engine...")
    remediator = Remediator({"lr": 0.01, "activation": "ReLU"})
    severity = hyps[0].confidence if hyps else 0.5
    new_config, info = remediator.remediate(hyps, severity=severity)
    print(f"    Fix: {info[:80]}")

    # Phase 3: Apply fix
    print("  [3/4] FIX — Retraining with corrected hyperparameters...")
    model2 = RNNDemo(hidden=32, num_layers=2)
    model_ref = [model2]

    events2, hyps2, chains2 = train_with_dbg(model2, data_fn, steps=steps, bug_fn=bug_fn,
                                              lr=new_config.get("lr", 0.01))
    n_after = 0
    for e in events2:
        et = e.event_type.value if hasattr(e, 'event_type') else e.get('event_type', '')
        ts = (e.to_state if hasattr(e, 'to_state') else e.get('to_state', '')).lower() if hasattr(e, 'to_state') or isinstance(e, dict) else ''
        if et == 'activation_regime_shift' and ts in ('normal', 'healthy', 'none', ''):
            continue
        n_after += 1

    improved = n_after < n_anomalies
    print(f"    After fix: {n_anomalies} -> {n_after} anomalies | Improved: {'YES' if improved else 'NO'}")

    # Phase 4: Validate
    print("  [4/4] VALIDATE — Checking fix quality...")
    if improved:
        status = "PASS"
        detail = f"Fix reduced anomalies by {n_anomalies - n_after} ({100*(n_anomalies-n_after)//max(n_anomalies,1)}%)"
    elif n_after == n_anomalies:
        status = "NOOP"
        detail = "Fix had no effect — bug too severe or wrong diagnosis"
    else:
        status = "FAIL"
        detail = f"Anomalies increased: {n_anomalies} -> {n_after}"

    print(f"    Status: {status} | {detail}")
    
    # Export Aquarium-compatible JSON
    export = {
        "events": [{k: v for k, v in e.items() if k != '_sa_instance_state'} if isinstance(e, dict) else {
            "event_type": e.event_type.value, "layer_name": e.layer_name,
            "step": e.step, "from_state": e.from_state, "to_state": e.to_state,
            "confidence": e.confidence
        } for e in events],
        "causal_chains": [{"root_cause": c.root_cause, "final_symptom": c.final_symptom,
                           "confidence": c.confidence, "nodes": len(c.links)}
                          for c in chains] if chains else [],
        "hypotheses": [{"description": h.description, "confidence": h.confidence} for h in hyps] if hyps else [],
        "loss_history": [],
        "meta": {"arch": "LSTM", "family": "RNN", "steps": steps, "bug": bug_name}
    }
    json_path = f"aquarium_export_{bug_name.replace(' ','_').replace('(','').replace(')','')}.json"
    with open(json_path, 'w') as f:
        json.dump(export, f, indent=2, default=str)
    print(f"    Exported: {json_path} (drag into docs/aquarium.html)")
    
    return {"bug": bug_name, "status": status, "before": n_anomalies, "after": n_after, "chain": top_chain}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("NEURALSUITE E2E RNN PIPELINE")
    print("Detect -> Diagnose -> Fix -> Validate on LSTM")
    print("=" * 55)

    results = []

    # Test exploding gradients
    results.append(pipeline("Exploding LR", bug_exploding))

    # Test NaN data
    results.append(pipeline("NaN in data", bug_nan_data))

    # Test zero init
    results.append(pipeline("Zero init", bug_zero_init))

    # Test vanishing (LSTM-specific)
    results.append(pipeline("Vanishing (forget gate)", bug_vanishing_rnn))

    # Summary
    print(f"\n{'='*55}")
    print("RESULTS")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['bug']:25s} | {r['status']:5s} | {r['before']:2d} -> {r['after']:2d} | {r['chain'][:45]}")

    passed = sum(1 for r in results if r['status'] == 'PASS')
    print(f"\n  Pipeline success: {passed}/{len(results)} RNN bugs auto-fixed")
    print(f"  NeuralSuite closed loop: DETECT -> DIAGNOSE -> FIX -> VALIDATE")
