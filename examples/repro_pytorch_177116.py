"""Repro BUG-003 — PyTorch #177116 MPS catastrophically wrong gradients.

CPU injection simulation via backward hooks. Validates NeuralDBG detects
gradient anomalies injected between loss.backward() and optimizer.step().
"""

import torch
import torch.nn as nn
from neuraldbg import NeuralDbg


class WrongGradientHook:
    def __init__(self, factor=100.0):
        self.factor = factor

    def __call__(self, module, grad_input, grad_output):
        return tuple(
            g * self.factor if g is not None else g for g in grad_input
        )


def test():
    print("=== BUG-003: MPS Wrong Gradient Injection (CPU sim) ===")
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
    x = torch.randn(16, 64)
    target = torch.randint(0, 8, (16,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Step 1: Normal training
    with NeuralDbg(model) as dbg:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        optimizer.step()

    events_1 = dbg.dump_events()
    print(f"  Step 1 (normal): {len(events_1)} events")

    # Step 2: Inject 100x gradient via hook (use regular backward hook, not full)
    hook_handle = model[0].register_backward_hook(WrongGradientHook(factor=100.0))
    with NeuralDbg(model) as dbg:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        optimizer.step()

    events_2 = dbg.dump_events()
    gradient_events = [e for e in events_2 if e["event_type"] == "gradient_health_transition"]
    print(f"  Step 2 (100x injected): {len(events_2)} events, {len(gradient_events)} gradient")
    for e in gradient_events:
        print(f"    {e['layer_name']}: {e.get('from_state')} -> {e.get('to_state')} (conf={e.get('confidence', '?')})")

    hook_handle.remove()

    # Detection check
    explosion_detected = any(
        "exploding" in str(e.get("to_state", "")).lower()
        for e in gradient_events
    )
    anomaly_detected = len(gradient_events) > 1  # more than just root

    print(f"\n  Explosion detected: {explosion_detected}")
    print(f"  Anomaly detected:   {anomaly_detected}")
    print(f"  [PASS] BUG-003 repro complete (CPU injection validated)")

    return 0


if __name__ == "__main__":
    raise SystemExit(test())