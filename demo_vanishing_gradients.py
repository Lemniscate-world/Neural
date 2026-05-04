#!/usr/bin/env python3
"""
Demonstration script showing NeuralDBG's causal inference for vanishing gradients.

This script creates a training scenario that leads to vanishing gradients and
demonstrates how the reframed NeuralDBG provides structured explanations.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _model_arch_str(model):
    """Return a compact representation of the model architecture."""
    return " -> ".join(type(module).__name__ for module in model.children()) or type(model).__name__


def _serialize_semantic_event(event):
    """Convert a SemanticEvent into an MLflow-friendly dictionary."""
    return {
        "event_type": event.event_type.value,
        "layer_name": event.layer_name,
        "step": event.step,
        "from_state": str(event.from_state),
        "to_state": str(event.to_state),
        "confidence": event.confidence,
        "metadata": event.metadata,
    }


def _serialize_causal_hypothesis(hypothesis):
    """Convert a CausalHypothesis into an MLflow-friendly dictionary."""
    return {
        "description": hypothesis.description,
        "confidence": hypothesis.confidence,
        "causal_chain": hypothesis.causal_chain,
        "evidence": [_serialize_semantic_event(event) for event in hypothesis.evidence],
    }

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
    X = torch.randn(1000, 10) * 0.1
    X.requires_grad_(True)
    y = torch.randn(1000, 1) * 0.01
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_with_monitoring(model, dataloader, num_steps=100):
    """
    Train the model while monitoring with NeuralDBG.

    This demonstrates the causal inference approach:
    1. Semantic events are extracted automatically via hooks.
    2. Loss values are recorded for optimizer instability detection.
    3. After training, the engine generates ranked causal hypotheses.
    """
    LR = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    mlflow_active = False
    mlflow_created_run = False
    tracking_uri = None

    if MLFLOW_AVAILABLE:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("neuraldbg-vanishing-gradients")
        if mlflow.active_run() is None:
            mlflow.start_run(run_name="demo_run")
            mlflow_created_run = True
        mlflow.log_params({
            "lr": LR,
            "batch_size": 32,
            "num_steps": num_steps,
            "threshold_vanishing": 1e-3,
            "model_arch": _model_arch_str(model),
            "seed": 42,
        })
        mlflow_active = True

    print("[TRAINING] NeuralDBG monitoring active...")
    print("   Model: Deep Tanh network with small LR")
    print("   Expected: Vanishing gradients due to saturation + small LR")
    if not MLFLOW_AVAILABLE:
        print("   MLflow unavailable: install mlflow to persist metrics and artifacts")
    print()

    with NeuralDbg(model, threshold_vanishing=1e-3) as dbg:
        for step in range(num_steps):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                dbg.step = step

                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()

                if mlflow_active:
                    mlflow.log_metric("loss", loss.item(), step=step)
                    for layer_name, grad_norm in dbg.previous_gradient_norms.items():
                        mlflow.log_metric(
                            "grad_norm/" + layer_name.replace(".", "_"),
                            grad_norm,
                            step=step,
                        )

                dbg.record_loss(loss.item())
                optimizer.step()
                break

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
    print("  - No data anomaly detection (NaN/Inf silently propagate)")
    print()
    print("NeuralDBG Approach:")
    print("  - Semantic events only (lightweight)")
    print("  - Automatic causal hypothesis generation")
    print("  - Root cause identification without debugging")
    print("  - Structured explanations with confidence scores")
    print("  - Data anomaly detection (NaN, Inf, distribution shifts)")
    print("  - Optimizer instability tracking (plateaus, spikes, divergence)")
    print()

    print("[ANALYSIS] Post-mortem Causal Analysis:")
    print("=" * 50)

    hypotheses = dbg.explain_failure("vanishing_gradients")

    if hypotheses:
        print(f"[RESULT] Found {len(hypotheses)} causal hypotheses:")
        for i, hyp in enumerate(hypotheses, 1):
            print(f"\n{i}. {hyp.description}")
            print(f"   Confidence: {hyp.confidence:.2f}")
            print(f"   Evidence: {len(hyp.evidence)} events")
            if hyp.causal_chain:
                print("   Chain:")
                for s in hyp.causal_chain:
                    print(f"     - {s}")
    else:
        print("[WARNING] No vanishing gradient events detected")

    couplings = dbg.detect_coupled_failures()
    if couplings:
        print(f"\n[COUPLING] Detected {len(couplings)} coupled failure patterns:")
        for coupling in couplings:
            trigger = coupling.get("trigger", coupling.get("event1", "unknown"))
            consequence = coupling.get("consequence", coupling.get("event2", "unknown"))
            print(f"   {trigger} <-> {consequence} (confidence: {coupling['confidence']:.2f})")

    print(f"\n[STATS] Total semantic events captured: {len(dbg.events)}")
    event_counts = {}
    for event in dbg.events:
        event_type = event.event_type.value
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

    for event_type, count in event_counts.items():
        print(f"   - {event_type}: {count} events")

    opt_hypotheses = dbg.explain_failure("optimizer_instability")
    if opt_hypotheses:
        print(f"\n[OPTIMIZER] Found {len(opt_hypotheses)} optimizer hypotheses:")
        for i, hyp in enumerate(opt_hypotheses, 1):
            print(f"   {i}. {hyp.description} (confidence: {hyp.confidence:.2f})")

    data_hypotheses = dbg.explain_failure("data_anomaly")
    if data_hypotheses:
        print(f"\n[DATA] Found {len(data_hypotheses)} data anomaly hypotheses:")
        for i, hyp in enumerate(data_hypotheses, 1):
            print(f"   {i}. {hyp.description} (confidence: {hyp.confidence:.2f})")

    collapsed = dbg._collapse_events()
    print(f"\n[COLLAPSED] {len(dbg.events)} raw events -> "
          f"{len(collapsed)} collapsed events")

    print("\n[GRAPH] Causal Graph (Mermaid):")
    print("-" * 50)
    print(dbg.export_mermaid_causal_graph())
    print("-" * 50)

    if mlflow_active:
        mlflow.log_dict(
            [_serialize_semantic_event(event) for event in dbg.events],
            "artifacts/semantic_events.json",
        )
        mlflow.log_dict(
            [_serialize_causal_hypothesis(hypothesis) for hypothesis in hypotheses],
            "artifacts/causal_hypotheses.json",
        )
        mlflow.log_text(dbg.export_mermaid_causal_graph(), "artifacts/causal_graph.mmd")
        mlflow.log_metrics({
            "total_events": float(len(dbg.events)),
            "unique_event_types": float(len({event.event_type.value for event in dbg.events})),
            "top_hypothesis_confidence": hypotheses[0].confidence if hypotheses else 0.0,
        })
        if mlflow_created_run:
            mlflow.end_run()

        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri.startswith("http"):
            print(f"\n[MLFLOW] Dashboard: {tracking_uri}")
        else:
            print(
                f"\n[MLFLOW] Run logged. View: mlflow ui "
                f"--backend-store-uri {tracking_uri} --port 5001"
            )

    return hypotheses

def main():
    """Main demonstration function."""
    print("[NeuralDBG] Causal Inference Demo")
    print("=" * 50)
    print()

    torch.manual_seed(42)

    model = create_failing_model()
    dataloader = create_problematic_data()

    print("[SETUP] Problem Setup:")
    print("   - Deep network with Tanh activations (prone to saturation)")
    print("   - Very small learning rate (0.0001)")
    print("   - Small input/target scales")
    print("   - Expected outcome: Vanishing gradients from LR x saturation mismatch")
    print()

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
