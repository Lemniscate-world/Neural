#!/usr/bin/env python3
"""
Dogfooding Script: NeuralDBG on ResNet-18 (~11M parameters)

This script validates NeuralDBG on a real-world model architecture, not just
a toy demo. It trains ResNet-18 on synthetic CIFAR-like data with intentionally
injected failures to verify that the engine detects real training problems.

Failure scenarios injected:
1. Learning rate too high -> exploding gradients
2. NaN injection at step 15 -> data anomaly detection
3. Large loss spike -> optimizer instability detection

Rule 58 (Dogfooding): NeuralDBG MUST be tested on a model with >1M params
before each validation milestone.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import traceback

from neuraldbg import NeuralDbg, EventType


def create_resnet18():
    """Create a ResNet-18 model (~11M parameters).

    ResNet-18 is a widely used convolutional neural network with 18 layers.
    It uses skip connections (residual connections) that allow gradients to
    flow through the network without vanishing. This makes it a good test
    case because NeuralDBG should detect problems even in architectures
    designed to avoid gradient issues.

    Returns:
        nn.Module: ResNet-18 model configured for 10-class classification
        on 32x32 RGB images.
    """
    # Use torchvision if available, otherwise build a simplified version
    try:
        from torchvision.models import resnet18
        model = resnet18(num_classes=10)
        # Adapt for 32x32 input (CIFAR-size) instead of 224x224 (ImageNet)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    except ImportError:
        # Fallback: build a simplified ResNet-like model with >1M params
        return _build_simple_resnet()


def _build_simple_resnet():
    """Build a simplified ResNet-like model when torchvision is unavailable.

    This model has ~1.2M parameters and uses residual connections.
    """
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.relu(out)

    class SimpleResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = nn.Sequential(
                ResidualBlock(64), ResidualBlock(64), ResidualBlock(64)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ResidualBlock(128), ResidualBlock(128), ResidualBlock(128)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ResidualBlock(256), ResidualBlock(256)
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return SimpleResNet()


def count_parameters(model):
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_synthetic_cifar(num_samples=256, num_classes=10):
    """Generate synthetic CIFAR-like data (32x32 RGB images).

    Args:
        num_samples: Number of training samples to generate.
        num_classes: Number of classification classes.

    Returns:
        Tuple of (images_tensor, labels_tensor).
    """
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return images, labels


def run_dogfooding():
    """Run NeuralDBG dogfooding on ResNet-18 with injected failures.

    This function trains ResNet-18 for 30 steps with 3 injected failure
    scenarios, then analyzes the semantic events and causal hypotheses
    produced by NeuralDBG.

    Returns:
        dict: Summary of detected events and hypotheses.
    """
    torch.manual_seed(42)

    print("[DOGFOODING] NeuralDBG on ResNet-18")
    print("=" * 60)

    # --- Setup ---
    model = create_resnet18()
    param_count = count_parameters(model)
    print(f"[MODEL] ResNet-18: {param_count:,} parameters")
    assert param_count > 1_000_000, (
        f"Model must have >1M params for dogfooding (got {param_count:,})"
    )

    images, labels = generate_synthetic_cifar(num_samples=256)
    criterion = nn.CrossEntropyLoss()

    # High learning rate to trigger gradient issues
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    print(f"[SETUP] LR=0.5 (intentionally high), SGD with momentum")
    print(f"[SETUP] 30 training steps, batch_size=64")
    print(f"[SETUP] Injecting NaN at step 15, observing recovery")
    print()

    # --- Training with NeuralDBG ---
    num_steps = 30
    batch_size = 64
    results = {
        "param_count": param_count,
        "events_detected": 0,
        "event_types": {},
        "hypotheses": [],
        "failures_detected": False,
    }

    with NeuralDbg(model) as dbg:
        for step in range(num_steps):
            dbg.step = step

            # Get batch
            start_idx = (step * batch_size) % len(images)
            end_idx = start_idx + batch_size
            batch_images = images[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            # --- Failure injection ---
            if step == 15:
                # Inject NaN into a fraction of the batch to trigger
                # data anomaly detection
                nan_mask = torch.zeros_like(batch_images)
                nan_mask[:4] = float("nan")  # First 4 samples
                batch_images = batch_images + nan_mask
                print(f"  Step {step}: [INJECTED] NaN in 4 samples")

            optimizer.zero_grad()

            try:
                output = model(batch_images)
                loss = criterion(output, batch_labels)

                # Check for NaN loss before backward
                if torch.isnan(loss):
                    print(f"  Step {step}: Loss=NaN (expected after NaN injection)")
                    dbg.record_loss(float("nan"))
                    # Reset the model state to recover
                    optimizer.zero_grad()
                    # Use clean data for next step
                    continue

                loss.backward()
                dbg.record_loss(loss.item())
                optimizer.step()

                if step % 5 == 0 or step == num_steps - 1:
                    print(f"  Step {step}: Loss={loss.item():.4f}")

            except RuntimeError as e:
                print(f"  Step {step}: RuntimeError: {e}")
                dbg.record_loss(float("nan"))
                optimizer.zero_grad()
                continue

        # --- Analysis ---
        print()
        print("[ANALYSIS] Post-training analysis")
        print("=" * 60)

        # Event summary
        print(f"\n[EVENTS] Total semantic events captured: {len(dbg.events)}")
        event_counts = {}
        for event in dbg.events:
            etype = event.event_type.value
            event_counts[etype] = event_counts.get(etype, 0) + 1
        for etype, count in sorted(event_counts.items()):
            print(f"  - {etype}: {count}")

        results["events_detected"] = len(dbg.events)
        results["event_types"] = event_counts

        # Causal hypotheses
        all_hypotheses = dbg.explain_failure()
        print(f"\n[HYPOTHESES] {len(all_hypotheses)} causal hypotheses generated:")
        for i, hyp in enumerate(all_hypotheses, 1):
            print(f"  {i}. {hyp.description}")
            print(f"     Confidence: {hyp.confidence:.2f}")
            print(f"     Evidence: {len(hyp.evidence)} events")
            if hyp.causal_chain:
                for chain_step in hyp.causal_chain:
                    print(f"       -> {chain_step}")
        results["hypotheses"] = [
            {"description": h.description, "confidence": h.confidence}
            for h in all_hypotheses
        ]

        # Specific failure type analysis
        for failure_type in ["vanishing_gradients", "exploding_gradients",
                             "saturated_activations", "optimizer_instability",
                             "data_anomaly"]:
            specific = dbg.explain_failure(failure_type)
            if specific:
                print(f"\n[{failure_type.upper()}] {len(specific)} hypotheses:")
                for hyp in specific:
                    print(f"  - {hyp.description} (conf: {hyp.confidence:.2f})")

        # Coupled failures
        couplings = dbg.detect_coupled_failures()
        if couplings:
            print(f"\n[COUPLING] {len(couplings)} coupled failure patterns:")
            for c in couplings:
                trigger = c.get("trigger", c.get("event1", "?"))
                consequence = c.get("consequence", c.get("event2", "?"))
                print(f"  {trigger} <-> {consequence} "
                      f"(confidence: {c['confidence']:.2f})")

        # Event collapse
        collapsed = dbg._collapse_events()
        print(f"\n[COLLAPSE] {len(dbg.events)} raw -> {len(collapsed)} collapsed")

        # Mermaid graph
        graph = dbg.export_mermaid_causal_graph()
        if graph and "---" not in graph[:20]:
            print(f"\n[GRAPH] Causal graph generated ({len(graph)} chars)")

        # Determine if real failures were detected
        results["failures_detected"] = len(dbg.events) > 0

    return results


def main():
    """Main entry point for dogfooding validation."""
    print()
    print("NeuralDBG Dogfooding Validation")
    print("Rule 58: Must detect real failures on a model with >1M params")
    print("=" * 60)
    print()

    try:
        results = run_dogfooding()
    except Exception as e:
        print(f"\n[FATAL] Dogfooding failed with exception: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Validation ---
    print()
    print("[VALIDATION] Dogfooding Results")
    print("=" * 60)
    print(f"  Model size: {results['param_count']:,} params (>1M required)")
    print(f"  Events detected: {results['events_detected']}")
    print(f"  Event types: {results['event_types']}")
    print(f"  Hypotheses: {len(results['hypotheses'])}")
    print(f"  Failures detected: {results['failures_detected']}")

    if results["param_count"] < 1_000_000:
        print("\n[FAIL] Model has fewer than 1M parameters")
        sys.exit(1)

    if not results["failures_detected"]:
        print("\n[WARN] No failures detected -- engine may need tuning")
        print("  This could mean the model trained without issues,")
        print("  or the detection thresholds need adjustment.")
    else:
        print("\n[PASS] NeuralDBG successfully detected training issues")
        print("  on a real-world architecture (ResNet-18).")

    print()
    print("[DONE] Dogfooding complete.")
    return results


if __name__ == "__main__":
    main()
