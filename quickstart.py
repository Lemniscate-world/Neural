#!/usr/bin/env python3
"""
NeuralDBG Quickstart — See the magic in 30 seconds.

Run: python quickstart.py
"""

import torch
import torch.nn as nn
from neuraldbg import NeuralDbg

# 1. Create a tiny model that will fail (vanishing gradients)
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

# Sabotage the weights to force a failure
with torch.no_grad():
    for param in model.parameters():
        param.fill_(1e-8)  # Tiny weights -> Vanishing gradients

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 2. Wrap your training loop with NeuralDbg
print("[Search] Training a sabotaged model with NeuralDBG...")
with NeuralDbg(model) as dbg:
    for step in range(5):
        optimizer.zero_grad()
        dbg.step = step
        x, y = torch.randn(4, 10), torch.randn(4, 1)
        loss = criterion(model(x), y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# 3. Get the explanation
print("\n[Analysis] NeuralDBG Analysis:")
hypotheses = dbg.explain_failure()
if hypotheses:
    for h in hypotheses:
        print(f"  - [{h.confidence:.0%}] {h.description}")
else:
    print("  (No specific failure detected, but events were captured)")

print("\n[OK] Done! Check the 'neuraldbg_report.json' for the full report.")
dbg.export_aquarium_package("neuraldbg_report.json")
