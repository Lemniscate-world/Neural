#!/usr/bin/env python3
"""
Demo: NeuralDBG Data Anomaly and Optimizer Instability Detection

This is the second demo for NeuralDBG (Rule 16: minimum 2 working demos).
While the first demo (demo_vanishing_gradients.py) focuses on gradient
health transitions, this demo showcases:

1. Data Anomaly Detection:
   - NaN values appearing in training data
   - Inf values from numerical overflow
   - Distribution shifts (data statistics change mid-training)

2. Optimizer Instability Detection:
   - Loss plateaus (training stalls)
   - Loss spikes (sudden jumps)
   - Divergence (loss goes to NaN/Inf)

3. Cross-domain Causal Reasoning:
   - How data anomalies cause optimizer instability
   - How the engine links events across different failure types

What is a "data anomaly"?
    In machine learning, your model learns from data. If that data suddenly
    contains garbage values (NaN = "Not a Number", Inf = infinity) or if
    the data distribution changes (e.g., images suddenly get 10x brighter),
    the model gets confused. NeuralDBG detects these problems automatically.

What is "optimizer instability"?
    The optimizer is the algorithm that adjusts model weights to reduce the
    loss (error). When the loss stops improving (plateau), jumps wildly
    (spike), or becomes NaN (divergence), the optimizer is unstable.
    NeuralDBG tracks these patterns and explains their likely causes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from neuraldbg import NeuralDbg, EventType


def create_simple_model():
    """Create a simple feedforward model for demonstrating data anomaly detection.

    This model is intentionally simple so the focus is on the data problems,
    not on the model architecture.

    Architecture: 20 -> 64 -> 64 -> 1 (regression)
    """
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def run_scenario_nan_injection():
    """Scenario 1: NaN values appear in training data.

    Simulates a common real-world problem: a data pipeline produces
    corrupted values (e.g., sensor failure, missing data filled with NaN).
    NeuralDBG should detect the NaN and report a DATA_ANOMALY event.
    """
    print("[SCENARIO 1] NaN Injection in Training Data")
    print("-" * 50)
    print("  Simulates: Corrupted data pipeline (sensor failure)")
    print("  Expected: DATA_ANOMALY event with NAN_DETECTED state")
    print()

    torch.manual_seed(42)
    model = create_simple_model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    with NeuralDbg(model) as dbg:
        for step in range(20):
            dbg.step = step

            # Normal data for first 8 steps
            x = torch.randn(32, 20)

            # Inject NaN at step 8
            if step == 8:
                x[0:5] = float("nan")  # 5 out of 32 samples corrupted
                print(f"  Step {step}: [INJECTED] NaN in 5 samples")

            # Inject clean data again at step 12 (recovery)
            if step == 12:
                print(f"  Step {step}: [RECOVERED] Clean data restored")

            y = torch.randn(32, 1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            if torch.isnan(loss):
                print(f"  Step {step}: Loss=NaN (NaN propagated through model)")
                dbg.record_loss(float("nan"))
                optimizer.zero_grad()
                continue

            loss.backward()
            dbg.record_loss(loss.item())
            optimizer.step()

            if step % 4 == 0:
                print(f"  Step {step}: Loss={loss.item():.4f}")

    # Analysis
    nan_events = [e for e in dbg.events
                  if e.event_type == EventType.DATA_ANOMALY]
    print(f"\n  Result: {len(nan_events)} data anomaly events detected")
    for e in nan_events:
        print(f"    Step {e.step}: {e.from_state} -> {e.to_state} "
              f"in '{e.layer_name}' (conf: {e.confidence:.2f})")

    hypotheses = dbg.explain_failure("data_anomaly")
    if hypotheses:
        print(f"  Hypotheses:")
        for h in hypotheses:
            print(f"    - {h.description} (conf: {h.confidence:.2f})")

    return dbg


def run_scenario_distribution_shift():
    """Scenario 2: Data distribution shifts mid-training.

    Simulates: Training data changes characteristics mid-stream.
    For example, in production ML, the input data might slowly drift
    (concept drift) or suddenly change (e.g., new data source).
    NeuralDBG should detect the statistical shift.
    """
    print("\n[SCENARIO 2] Distribution Shift in Training Data")
    print("-" * 50)
    print("  Simulates: Data source changes (concept drift)")
    print("  Expected: DATA_ANOMALY event with DISTRIBUTION_SHIFT state")
    print()

    torch.manual_seed(123)
    model = create_simple_model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    with NeuralDbg(model) as dbg:
        for step in range(25):
            dbg.step = step

            if step < 10:
                # Normal distribution: mean=0, std=1
                x = torch.randn(32, 20)
            else:
                # Shifted distribution: mean=5, std=3
                # This simulates a sudden change in data source
                x = torch.randn(32, 20) * 3 + 5

            if step == 10:
                print(f"  Step {step}: [SHIFTED] Data distribution "
                      f"changed from N(0,1) to N(5,3)")

            y = torch.randn(32, 1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            dbg.record_loss(loss.item())
            optimizer.step()

            if step % 5 == 0:
                print(f"  Step {step}: Loss={loss.item():.4f}")

    shift_events = [e for e in dbg.events
                    if e.event_type == EventType.DATA_ANOMALY
                    and "distribution_shift" in str(e.to_state)]
    print(f"\n  Result: {len(shift_events)} distribution shift events detected")
    for e in shift_events:
        print(f"    Step {e.step}: {e.from_state} -> {e.to_state} "
              f"in '{e.layer_name}'")

    # Show all events for context
    print(f"  Total events: {len(dbg.events)}")
    event_types = {}
    for e in dbg.events:
        t = e.event_type.value
        event_types[t] = event_types.get(t, 0) + 1
    for t, c in sorted(event_types.items()):
        print(f"    - {t}: {c}")

    return dbg


def run_scenario_optimizer_instability():
    """Scenario 3: Optimizer instability from bad hyperparameters.

    Simulates: Learning rate too high causes loss spikes and divergence.
    This is one of the most common training failures in practice.
    NeuralDBG should detect the loss spike and divergence events.
    """
    print("\n[SCENARIO 3] Optimizer Instability (High Learning Rate)")
    print("-" * 50)
    print("  Simulates: LR too high -> loss spike -> divergence")
    print("  Expected: OPTIMIZER_INSTABILITY events (spike, diverging)")
    print()

    torch.manual_seed(99)
    model = create_simple_model()
    # Intentionally extreme learning rate
    optimizer = optim.SGD(model.parameters(), lr=10.0)
    criterion = nn.MSELoss()

    with NeuralDbg(model) as dbg:
        for step in range(15):
            dbg.step = step

            x = torch.randn(32, 20)
            y = torch.randn(32, 1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Step {step}: Loss={'NaN' if torch.isnan(loss) else 'Inf'} "
                      f"(model diverged)")
                dbg.record_loss(float("nan") if torch.isnan(loss) else float("inf"))
                break

            loss.backward()
            dbg.record_loss(loss.item())
            optimizer.step()

            print(f"  Step {step}: Loss={loss.item():.4f}")

    opt_events = [e for e in dbg.events
                  if e.event_type == EventType.OPTIMIZER_INSTABILITY]
    print(f"\n  Result: {len(opt_events)} optimizer instability events detected")
    for e in opt_events:
        print(f"    Step {e.step}: {e.from_state} -> {e.to_state} "
              f"(conf: {e.confidence:.2f})")

    hypotheses = dbg.explain_failure("optimizer_instability")
    if hypotheses:
        print(f"  Hypotheses:")
        for h in hypotheses:
            print(f"    - {h.description} (conf: {h.confidence:.2f})")

    return dbg


def run_scenario_cross_domain():
    """Scenario 4: Cross-domain causal reasoning.

    Demonstrates how NeuralDBG links events across different failure types.
    A NaN in the data causes gradient explosion which causes optimizer
    divergence -- the engine should identify the causal chain.
    """
    print("\n[SCENARIO 4] Cross-Domain Causal Chain")
    print("-" * 50)
    print("  Simulates: NaN data -> gradient explosion -> optimizer divergence")
    print("  Expected: Events across DATA_ANOMALY, GRADIENT, and OPTIMIZER")
    print()

    torch.manual_seed(77)
    model = create_simple_model()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    with NeuralDbg(model) as dbg:
        for step in range(20):
            dbg.step = step

            x = torch.randn(32, 20)

            # Inject Inf at step 6 to trigger cascading failure
            if step == 6:
                x[0:3] = float("inf")
                print(f"  Step {step}: [INJECTED] Inf in 3 samples")

            y = torch.randn(32, 1)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)

            if torch.isnan(loss) or torch.isinf(loss):
                val = "NaN" if torch.isnan(loss) else "Inf"
                print(f"  Step {step}: Loss={val} (cascading failure)")
                dbg.record_loss(
                    float("nan") if torch.isnan(loss) else float("inf")
                )
                optimizer.zero_grad()
                continue

            loss.backward()
            dbg.record_loss(loss.item())
            optimizer.step()

            if step % 3 == 0:
                print(f"  Step {step}: Loss={loss.item():.4f}")

    # Full analysis
    print(f"\n  Total events: {len(dbg.events)}")
    event_types = {}
    for e in dbg.events:
        t = e.event_type.value
        event_types[t] = event_types.get(t, 0) + 1
    for t, c in sorted(event_types.items()):
        print(f"    - {t}: {c}")

    # Cross-domain coupling
    couplings = dbg.detect_coupled_failures()
    if couplings:
        print(f"\n  Coupled failures detected: {len(couplings)}")
        for c in couplings:
            trigger = c.get("trigger", c.get("event1", "?"))
            consequence = c.get("consequence", c.get("event2", "?"))
            print(f"    {trigger} <-> {consequence} "
                  f"(conf: {c['confidence']:.2f})")

    # Causal graph
    graph = dbg.export_mermaid_causal_graph()
    if graph:
        print(f"\n  Causal graph ({len(graph)} chars):")
        # Show first few lines
        for line in graph.split("\n")[:8]:
            print(f"    {line}")
        if graph.count("\n") > 8:
            print(f"    ... ({graph.count(chr(10)) - 8} more lines)")

    return dbg


def main():
    """Run all data anomaly and optimizer instability scenarios."""
    print()
    print("NeuralDBG Demo: Data Anomaly and Optimizer Instability")
    print("=" * 60)
    print()
    print("This demo shows NeuralDBG detecting problems in training DATA")
    print("(not just gradients). These are common real-world failures:")
    print("  - Corrupted data (NaN/Inf from bad pipelines)")
    print("  - Distribution shifts (data changes mid-training)")
    print("  - Optimizer divergence (bad hyperparameters)")
    print("  - Cascading failures (data -> gradients -> optimizer)")
    print()

    dbg1 = run_scenario_nan_injection()
    dbg2 = run_scenario_distribution_shift()
    dbg3 = run_scenario_optimizer_instability()
    dbg4 = run_scenario_cross_domain()

    # Summary
    print()
    print("=" * 60)
    print("[SUMMARY] All 4 scenarios completed")
    print("=" * 60)
    total_events = sum(
        len(d.events) for d in [dbg1, dbg2, dbg3, dbg4]
    )
    print(f"  Total events across all scenarios: {total_events}")
    print()
    print("Key takeaways:")
    print("  1. NeuralDBG detects data corruption automatically")
    print("  2. Distribution shifts are caught via statistical tracking")
    print("  3. Optimizer instability is linked to root causes")
    print("  4. Cross-domain reasoning connects data -> gradient -> optimizer")
    print()
    print("[DONE] Demo complete.")


if __name__ == "__main__":
    main()
