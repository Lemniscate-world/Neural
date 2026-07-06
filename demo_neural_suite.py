"""
NeuralSuite End-to-End Demo — Realistic Training Failure + Optimization

Demonstrates the full stack on a realistic CNN model:
  1. NeuralDBG detects training bugs in real-time
  2. NeuralPrune finds redundancy/optimization opportunities
  3. Tier 3 predictive detector flags anomalies vs healthy baseline
  4. Causal chain extraction explains the failure

Usage: python demo_neural_suite.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraldbg import NeuralDbg
from neuraldbg.prune import NeuralPrune


# ---------------------------------------------------------------------------
# Realistic model: Small CNN for CIFAR-like classification
# ---------------------------------------------------------------------------

class RealisticCNN(nn.Module):
    """A CNN with some deliberately suboptimal design choices:
    - One layer with too many filters (redundant)
    - Sigmoid activation on one branch (vanishing risk)
    - No BatchNorm on early layers (instability risk)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Early conv: no BatchNorm (risky)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Over-parameterized layer (128 filters for 8x8 feature map = redundant)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Branch with Sigmoid (vanishing risk)
        self.conv4a = nn.Conv2d(128, 64, 1)
        self.conv4b = nn.Conv2d(128, 64, 1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # Early pooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Sigmoid branch — WILL saturate and cause vanishing
        branch_a = torch.sigmoid(self.conv4a(x))
        branch_b = F.relu(self.conv4b(x))
        x = branch_a + branch_b
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  NeuralSuite End-to-End Demo")
    print("  NeuralDBG + NeuralPrune + Tier 3 Predictive Detector")
    print("=" * 65)
    
    model = RealisticCNN(num_classes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"\nModel: RealisticCNN ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Device: {device}")
    print(f"Design issues: Sigmoid branch, over-parameterized conv3, no BN on conv1")
    
    # ------------------------------------------------------------------
    # Phase 1: NeuralPrune — analyze redundancy BEFORE training
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("Phase 1: NeuralPrune — Pre-training redundancy analysis")
    print(f"{'─'*50}")
    
    pruner = NeuralPrune(model, warmup_steps=20)
    
    # Warmup: run a few forward/backward passes to collect stats
    print("  Warming up (20 steps)...")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for i in range(25):
        x = torch.randn(8, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        pruner.step()
        opt.step()
    
    prune_report = pruner.analyze()
    print(f"\n  {prune_report.summary}")
    print(f"\n  Top recommendations:")
    for rec in prune_report.recommendations[:5]:
        print(f"    [{rec.confidence:.0%}] {rec.signal.value:20s} | {rec.layer_name:25s} | {rec.suggested_action[:60]}")
    
    # ------------------------------------------------------------------
    # Phase 2: NeuralDBG — train with and without bugs
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("Phase 2: NeuralDBG — Training with injected bugs")
    print(f"{'─'*50}")
    
    # Re-create model for fresh training
    model2 = RealisticCNN(num_classes=10).to(device)
    
    with NeuralDbg(model2) as dbg:
        opt = torch.optim.SGD(model2.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # Healthy training (5 steps)
        print("  [Healthy] 5 steps...")
        for s in range(5):
            x = torch.randn(8, 3, 32, 32, device=device)
            y = torch.randint(0, 10, (8,), device=device)
            opt.zero_grad()
            loss = loss_fn(model2(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()
        
        healthy_events = len(dbg.dump_events())
        print(f"    Events: {healthy_events}")
        
        # Inject vanishing bug: scale down all weights 1000x
        print("  [Vanishing] 5 steps with weight decay...")
        with torch.no_grad():
            for p in model2.parameters():
                if p.dim() >= 2:
                    p.mul_(0.001)
        
        for s in range(10):
            x = torch.randn(8, 3, 32, 32, device=device)
            y = torch.randint(0, 10, (8,), device=device)
            opt.zero_grad()
            loss = loss_fn(model2(x), y)
            loss.backward()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            opt.step()
        
        bug_events = dbg.dump_events()
        chains = dbg.explain_causal()
        
        print(f"    Events: {len(bug_events)} (vs {healthy_events} healthy)")
        print(f"    Causal chains: {len(chains)}")
        
        # Show top chain
        if chains:
            top = chains[0]
            print(f"    Top chain: {top.root_cause} -> {top.final_symptom}")
        
        # Find vanishing events
        vanishing = [e for e in bug_events 
                     if 'vanishing' in str(e.get('to_state', '')).lower()
                     or 'vanishing' in str(e.get('event_type', '')).lower()]
        print(f"    Vanishing events: {len(vanishing)}")
        
        # Hypotheses
        hyps = dbg.explain_failure()
        print(f"    Hypotheses: {len(hyps)}")
        if hyps:
            print(f"    Best: {hyps[0].description[:100]}...")
    
    # ------------------------------------------------------------------
    # Phase 3: Tier 3 — Predictive anomaly detection
    # ------------------------------------------------------------------
    print(f"\n{'─'*50}")
    print("Phase 3: Tier 3 — Predictive anomaly detection")
    print(f"{'─'*50}")
    
    # Save buggy export
    export_path = "demo_export.json"
    with open(export_path, "w") as f:
        json.dump({"events": bug_events, "model": "RealisticCNN", 
                    "bug": "vanishing_weights"}, f, indent=2)
    
    # Run predictive detector
    try:
        from predictive_detector import detect_anomalies
        anomalies = detect_anomalies(export_path, family="CNN")
        print(f"  Family-aware (CNN): {len(anomalies)} anomalies")
        for a in anomalies:
            print(f"    {a['metric']}: z={a['z_score']:.1f} (value={a['value']:.3f}, baseline mean={a['mean']:.3f})")
        
        if not anomalies:
            print("  No anomalies — model is within normal CNN training range")
        else:
            print(f"  ⚠ BLACK SWAN DETECTED — {len(anomalies)} metrics outside normal range")
    except ImportError:
        print("  Tier 3 detector not available (run predictive_detector.py --train first)")
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  DEMO COMPLETE")
    print(f"{'='*65}")
    print(f"""
  NeuralPrune:  {len(prune_report.recommendations)} optimization recommendations
  NeuralDBG:    {len(vanishing)} vanishing events, {len(chains)} causal chains
  Tier 3:       {len(anomalies) if 'anomalies' in dir() else '?'} anomalies detected
  Export:       {export_path}
""")

if __name__ == "__main__":
    main()
