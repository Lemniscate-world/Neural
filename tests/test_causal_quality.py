import torch.nn as nn
from neuraldbg import NeuralDbg


def test_causal_quality():
    # 1. Create a model with Sequential to trigger digit names
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())

    dbg = NeuralDbg(model)

    # 2. Check names
    print("Layer Names Mapping:")
    for mod_id, name in dbg._module_names.items():
        print(f"  {mod_id}: {name}")

    # 3. Simulate some events close in time to check for duplicate couplings
    from neuraldbg import SemanticEvent, EventType

    dbg.events.append(
        SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name="Linear_0",
            step=1,
            from_state="NORMAL",
            to_state="SATURATED",
            confidence=1.0,
            metadata={},
        )
    )

    dbg.events.append(
        SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="Linear_2",
            step=2,
            from_state="HEALTHY",
            to_state="VANISHING",
            confidence=1.0,
            metadata={},
        )
    )

    # Add a redundant event to see if it causes duplicate couplings
    dbg.events.append(
        SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name="Linear_0",
            step=1,
            from_state="NORMAL",
            to_state="SATURATED",
            confidence=1.0,
            metadata={"extra": "redundant"},
        )
    )

    couplings = dbg.detect_coupled_failures(window=5)
    print(f"\nCouplings Detected: {len(couplings)}")
    for c in couplings:
        print(f"  {c['trigger']} -> {c['consequence']} (confidence: {c['confidence']})")

    # Check for duplicates
    trigger_consequences = [(c["trigger"], c["consequence"]) for c in couplings]
    if len(trigger_consequences) != len(set(trigger_consequences)):
        print("\n[FAIL] Duplicate couplings detected!")
    else:
        print("\n[PASS] No duplicate couplings.")


if __name__ == "__main__":
    test_causal_quality()
