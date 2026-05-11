import torch
import torch.nn as nn
import torch.optim as optim
from neuraldbg import NeuralDbg


def test_saturation_detection():
    # Simple model with Tanh (prone to saturation)
    model = nn.Sequential(nn.Linear(10, 50), nn.Tanh(), nn.Linear(50, 1))

    # Force saturation by setting very large weights in the first layer
    with torch.no_grad():
        model[0].weight.fill_(100.0)
        model[0].bias.fill_(100.0)

    # Initialize NeuralDbg
    dbg = NeuralDbg(model)

    # Mock training step
    x = torch.randn(16, 10)
    y = torch.ones(16, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print("Running training steps with forced saturation...")
    with dbg:
        for _ in range(2):
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dbg.step += 1

    # Check for saturation events
    print(f"Captured events: {len(dbg.events)}")
    for event in dbg.events:
        print(f"Event: {event.event_type} in {event.layer_name}, Step {event.step}")
        if isinstance(event.to_state, dict) and "saturation_ratio" in event.to_state:
            print(f"  Saturation Ratio: {event.to_state['saturation_ratio']:.4f}")

    # Get hypotheses
    hypotheses = dbg.explain_failure("saturated_activations")
    print("\nHypotheses for 'saturated_activations':")
    for h in hypotheses:
        print(f"- {h.description} (Confidence: {h.confidence:.2f})")
        print(f"  Causal Chain: {' -> '.join(h.causal_chain)}")

    assert len(hypotheses) > 0, "No saturation hypothesis generated!"
    print("\nVerification successful!")


if __name__ == "__main__":
    test_saturation_detection()
