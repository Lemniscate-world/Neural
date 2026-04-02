#!/usr/bin/env python3
"""
Demonstration script showing NeuralDBG's causal inference for vanishing gradients.

This script creates a training scenario that leads to vanishing gradients and
demonstrates how the reframed NeuralDBG provides structured explanations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg

def create_failing_model():
    """Create a model prone to vanishing gradients."""
    return nn.Sequential(
        nn.Linear(10, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 50),
        nn.Tanh(),
        nn.Linear(50, 20),
        nn.Tanh(),
        nn.Linear(20, 1)
    )

def create_problematic_data():
    """Create data that exacerbates vanishing gradients."""
    # Small learning rate with saturating activations
    X = torch.randn(1000, 10) * 0.1
    X.requires_grad_(True)  # Ensure hooks fire properly
    y = torch.randn(1000, 1) * 0.01
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_with_monitoring(model, dataloader, num_steps=100):
    """
    Train the model while monitoring with NeuralDBG.

    This demonstrates the causal inference approach.
    """
    optimizer = optim.SGD(model.parameters(), lr=0.0001)  # Even smaller LR
    criterion = nn.MSELoss()

    print("[TRAINING] NeuralDBG monitoring active...")
    print("   Model: Deep Tanh network with small LR")
    print("   Expected: Vanishing gradients due to saturation + small LR")
    print()

    with NeuralDbg(model, threshold_vanishing=1e-3) as dbg:  # More sensitive threshold for demo
        for step in range(num_steps):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                # Keep semantic event timestamps aligned with this training step.
                dbg.step = step

                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()

                optimizer.step()
                break  # Just one batch per step for demo

            if step % 20 == 0:
                print(f"Step {step}: Loss = {loss.item():.6f}")

    print()
    print("[COMPARISON] NeuralDBG vs Traditional Tools:")
    print("=" * 50)
    print()
    print("Traditional Approach (TensorBoard/WandB):")
    print("  - Stores full tensor histograms (memory heavy)")
    print("  - Shows gradient norms over time (passive monitoring)")
    print("  - Requires manual inspection to find patterns")
    print("  - No causal reasoning - just raw data visualization")
    print()
    print("NeuralDBG Approach:")
    print("  - Semantic events only (lightweight)")
    print("  - Automatic causal hypothesis generation")
    print("  - Root cause identification without debugging")
    print("  - Structured explanations with confidence scores")
    print()

    print("[ANALYSIS] Post-mortem Causal Analysis:")
    print("=" * 50)

    # Get causal explanations
    hypotheses = dbg.explain_failure("vanishing_gradients")

    if hypotheses:
        print(f"[RESULT] Found {len(hypotheses)} causal hypotheses:")
        for i, hyp in enumerate(hypotheses, 1):
            print(f"\n{i}. {hyp.description}")
            print(f"   Confidence: {hyp.confidence:.2f}")
            print(f"   Evidence: {len(hyp.evidence)} events")
            if hyp.causal_chain:
                print("   Chain:")
                for step in hyp.causal_chain:
                    print(f"     - {step}")
    else:
        print("[WARNING] No vanishing gradient events detected")

    # Show detected coupled failures
    couplings = dbg.detect_coupled_failures()
    if couplings:
        print(f"\n[COUPLING] Detected {len(couplings)} coupled failure patterns:")
        for coupling in couplings:
            trigger = coupling.get("trigger", coupling.get("event1", "unknown"))
            consequence = coupling.get("consequence", coupling.get("event2", "unknown"))
            print(f"   {trigger} <-> {consequence} (confidence: {coupling['confidence']:.2f})")

    # Show all semantic events detected
    print(f"\n[STATS] Total semantic events captured: {len(dbg.events)}")
    event_counts = {}
    for event in dbg.events:
        event_type = event.event_type.value
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    for event_type, count in event_counts.items():
        print(f"   - {event_type}: {count} events")

    # Show Mermaid graph
    print("\n[GRAPH] Causal Graph (Mermaid):")
    print("-" * 50)
    print(dbg.export_mermaid_causal_graph())
    print("-" * 50)

    return hypotheses

def main():
    """Main demonstration function."""
    print("[NeuralDBG] Causal Inference Demo")
    print("=" * 50)
    print()

    # Set random seed for reproducible failure
    torch.manual_seed(42)

    # Create the failing scenario
    model = create_failing_model()
    dataloader = create_problematic_data()

    print("[SETUP] Problem Setup:")
    print("   - Deep network with Tanh activations (prone to saturation)")
    print("   - Very small learning rate (0.0001)")
    print("   - Small input/target scales")
    print("   - Expected outcome: Vanishing gradients from LR x saturation mismatch")
    print()

    # Train and analyze
    hypotheses = train_with_monitoring(model, dataloader)

    print()
    print("[DONE] Demo Complete!")
    print()
    print("Key Insights:")
    print("- No tensor storage - only semantic events")
    print("- Causal hypotheses ranked by confidence")
    print("- Compiler-safe (module boundary monitoring)")
    print("- Abductive reasoning, not deductive inspection")

    if hypotheses:
        print("- Successfully identified root cause without debugging!")

if __name__ == "__main__":
    main()
