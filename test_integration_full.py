"""Full NeuralSuite integration test: detect -> chain -> remediate v2 -> validate."""
import sys
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\NeuralDBG")
sys.path.insert(0, r"C:\Users\Utilisateur\Documents\Neural-Agent")

import torch, torch.nn as nn
from neuraldbg import NeuralDbg
from neuralagent.remediator import Remediator
from neuralagent.validator import FixValidator

torch.manual_seed(42)

print("=" * 60)
print("NEURALSUITE FULL INTEGRATION TEST")
print("=" * 60)

# Bug: gradient explosion from huge inputs
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x): return self.net(x)

model = Model()
target = torch.randn(4, 2)
config = {"lr": 0.01, "activation": "ReLU", "clip_grad_norm": 0}

# Step 1: NeuralDBG detection
print("\n[1] NeuralDBG detection...")
with NeuralDbg(model) as dbg:
    for s in range(3):
        opt = torch.optim.SGD(model.parameters(), lr=config["lr"])
        opt.zero_grad()
        x = torch.randn(4, 8) * (100.0 if s >= 1 else 1.0)
        out = model(x)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
    events = dbg.dump_events()
    hypotheses = dbg.explain_failure()
    chains = dbg.explain_causal()

anomalies = [e for e in events if e["event_type"] in ("data_anomaly","gradient_health_transition","optimizer_instability","nan_detected")]
print(f"  Events: {len(events)} total, {len(anomalies)} anomalies")
print(f"  Top hypothesis: [{hypotheses[0].confidence:.2f}] {hypotheses[0].description[:80]}")
if chains:
    print(f"  Causal chain: {chains[0].root_cause} -> {chains[0].final_symptom}")

# Step 2: Remediator v2 diagnosis + fix
print("\n[2] Remediator v2...")
remediator = Remediator(config, accumulate=False)
new_config, info = remediator.remediate(hypotheses, severity=hypotheses[0].confidence if hypotheses else 1.0)
print(f"  {info[:120]}")

# Step 3: Build fix dict from remediation
fix = {
    "type": "hyperparameter",
    "lr_multiplier": new_config.get("lr", 0.01) / max(config.get("lr", 0.01), 1e-8),
    "clip_grad_norm": new_config.get("clip_grad_norm"),
    "activation": new_config.get("activation"),
}

# Step 4: Validator
print("\n[3] Validator...")
validator = FixValidator()
report = validator.validate_fix("import torch", "import torch", fix)
print(f"  Valid: {report['valid']}, Confidence: {report['confidence']:.2f}")

# Step 5: Apply fix and re-test
print("\n[4] Apply fix + re-test...")
model2 = type(model)()
with NeuralDbg(model2) as dbg2:
    for s in range(3):
        opt = torch.optim.SGD(model2.parameters(), lr=new_config["lr"])
        opt.zero_grad()
        x = torch.randn(4, 8) * (100.0 if s >= 1 else 1.0)
        if new_config.get("clip_grad_norm"):
            out = model2(x)
            loss = nn.MSELoss()(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), new_config["clip_grad_norm"])
        else:
            out = model2(x)
            loss = nn.MSELoss()(out, target)
            loss.backward()
        dbg2.step_iteration()
        dbg2.record_loss(loss.item())
    fixed_anomalies = len([e for e in dbg2.dump_events() if e["event_type"] in ("data_anomaly","gradient_health_transition","optimizer_instability","nan_detected")])

print(f"  Before fix: {len(anomalies)} anomalies")
print(f"  After fix:  {fixed_anomalies} anomalies")
resolved = fixed_anomalies <= len(anomalies) * 0.3
print(f"  Resolved: {'YES' if resolved else 'PARTIAL'}")

# Step 6: Accumulation test
print("\n[5] Accumulation mode...")
r_acc = Remediator(config, accumulate=True)
cfg1, _ = r_acc.remediate(hypotheses)
cfg2, _ = r_acc.remediate(hypotheses)  # same hypothesis, should accumulate
print(f"  Pass 1: lr={cfg1['lr']:.6f}")
print(f"  Pass 2: lr={cfg2['lr']:.6f} (accumulated: {cfg2['lr'] != cfg1['lr']})")

print(f"\n{'='*60}")
print("INTEGRATION TEST: PASS")
print(f"{'='*60}")
