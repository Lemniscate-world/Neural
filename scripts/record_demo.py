#!/usr/bin/env python3
"""
Script to run the NeuralDBG demo for recording.
Outputs clear, step-by-step messages perfect for a screen recording.
"""

import time
import sys
import torch
import torch.nn as nn
from neuraldbg import NeuralDbg


def print_step(msg):
    print(f"\n{'=' * 50}")
    print(f"👉 {msg}")
    print(f"{'=' * 50}")
    time.sleep(1.5)


def main():
    print("🎬 NeuralDBG Demo Recording Script")
    print("   Run this script while recording your screen for a clean demo video.")
    time.sleep(2)

    print_step("1. Creating a simple PyTorch model")
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    print("   Model: Linear(10->5) -> ReLU -> Linear(5->1)")
    time.sleep(1)

    print_step("2. Sabotaging weights to force vanishing gradients")
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1e-8)
    print("   Weights set to 1e-8 (near zero)")
    time.sleep(1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print_step("3. Wrapping training loop with NeuralDbg")
    print("   Code: with NeuralDbg(model) as dbg:")
    time.sleep(1)

    print_step("4. Running training loop (5 steps)")
    with NeuralDbg(model) as dbg:
        for step in range(5):
            optimizer.zero_grad()
            dbg.step = step
            x, y = torch.randn(4, 10), torch.randn(4, 1)
            loss = criterion(model(x), y)
            loss.backward()
            dbg.record_loss(loss.item())
            optimizer.step()
            print(f"   Step {step}: Loss = {loss.item():.6f}")
            time.sleep(0.5)

    print_step("5. Asking NeuralDBG for the diagnosis")
    print("   Code: hypotheses = dbg.explain_failure()")
    time.sleep(1)

    hypotheses = dbg.explain_failure()
    if hypotheses:
        print("\n📊 DIAGNOSIS:")
        for h in hypotheses:
            print(f"   - [{h.confidence:.0%}] {h.description}")
    else:
        print("\n   (No specific failure detected, but events were captured)")

    print_step("6. Exporting report to JSON")
    dbg.export_aquarium_package("demo_report.json")
    print("   Saved to demo_report.json")
    time.sleep(1)

    print("\n✅ Demo complete! You can now stop recording.")
    print("   Check 'demo_report.json' for the full structured output.")


if __name__ == "__main__":
    main()
