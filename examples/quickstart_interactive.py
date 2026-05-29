#!/usr/bin/env python3
"""
NeuralDBG Interactive Quickstart — Designed for Google Colab & Local Terminals.

To run in Google Colab, copy-paste this code and prepend:
!pip install neuraldbg

This script trains a deep MLP with a learning rate that is too high (causing exploding gradients)
or weights sabotaged (causing vanishing gradients), and demonstrates how NeuralDBG traces the
failure to its causal root cause.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from neuraldbg import NeuralDbg


def build_sabotaged_model(mode="vanishing"):
    # Create a deep MLP
    layers = []
    input_dim = 20
    for i in range(8):
        layers.append(nn.Linear(input_dim, 20))
        layers.append(nn.ReLU())
        input_dim = 20
    layers.append(nn.Linear(20, 1))
    model = nn.Sequential(*layers)

    if mode == "vanishing":
        # Sabotage weights to be tiny to force vanishing gradients in early layers
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1e-5)
    return model


def run_demo(mode="vanishing"):
    print(f"\n--- Running Scenario: {mode.upper()} GRADIENTS ---")

    # 1. Setup model and optimizer
    model = build_sabotaged_model(mode)
    lr = 0.01 if mode == "vanishing" else 1e5  # Extreme high LR for exploding
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 2. Wrap the model with NeuralDbg
    # We set custom thresholds for demonstration purposes
    with NeuralDbg(
        model, threshold_vanishing=1e-4, threshold_exploding=10.0
    ) as dbg:
        for step in range(10):
            optimizer.zero_grad()
            dbg.step = step

            # Dummy inputs
            x = torch.randn(8, 20)
            y = torch.randn(8, 1)

            # Forward and backward passes
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            # Record loss at each step to monitor optimizer instability
            dbg.record_loss(loss.item())
            optimizer.step()

            print(
                f"  Step {step + 1}/10 | Loss: {loss.item():.4f} | Max Grad Norm: {max(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.2e}"
            )

    # 3. Analyze causal hypotheses
    print("\n[NeuralDBG] Causal Reasoning Analysis:")
    hypotheses = dbg.explain_failure()

    if hypotheses:
        for i, h in enumerate(hypotheses, 1):
            print(f"\n  Hypothesis #{i} [Confidence: {h.confidence:.0%}]")
            print(f"    Description : {h.description}")
            print(f"    Causal Chain: {' -> '.join(h.causal_chain)}")
    else:
        print("  No critical failure detected (model remained stable).")

    # 4. Export report
    report_file = f"neuraldbg_report_{mode}.json"
    dbg.export_aquarium_package(report_file)
    print(f"\n[Export] Saved causal diagnostic package to: {report_file}")

    # 5. Output ASCII representation of the causal graph (Mermaid)
    print("\n[Visual] Generated Causal Graph (Mermaid syntax):")
    print(dbg.export_mermaid_causal_graph())


if __name__ == "__main__":
    torch.manual_seed(42)
    print("==================================================================")
    print("              NEURALDBG INTERACTIVE DEMO (Colab Ready)            ")
    print("==================================================================")

    # Choose scenario
    mode = "vanishing"
    if len(sys.argv) > 1 and sys.argv[1] in ["vanishing", "exploding"]:
        mode = sys.argv[1]

    run_demo(mode)
