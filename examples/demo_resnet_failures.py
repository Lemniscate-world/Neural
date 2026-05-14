#!/usr/bin/env python3
"""
ResNet-18 failure scenarios demonstrating NeuralDBG causal inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg


def _replace_activations(module, old_cls, new_cls):
    for name, child in module.named_children():
        if isinstance(child, old_cls):
            setattr(module, name, new_cls())
        else:
            _replace_activations(child, old_cls, new_cls)


def create_resnet18(activation="relu"):
    """Create ResNet-18 with optional Tanh activation override."""
    try:
        from torchvision.models import resnet18
    except ImportError:
        raise ImportError("torchvision required for ResNet demo")

    model = resnet18(weights=None, num_classes=10)
    if activation == "tanh":
        _replace_activations(model, nn.ReLU, nn.Tanh)
    return model


def _make_loader(num_samples, img_size):
    X = torch.randn(num_samples, 3, img_size, img_size)
    y = torch.randint(0, 10, (num_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=4, shuffle=True)


def train_resnet(
    model,
    dataloader,
    num_steps=20,
    lr=0.001,
    nan_step=None,
    threshold_vanishing=1e-6,
    threshold_exploding=1e3,
):
    """Train ResNet with NeuralDBG monitoring. Returns the dbg instance."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    nan_injected = False

    with NeuralDbg(
        model,
        threshold_vanishing=threshold_vanishing,
        threshold_exploding=threshold_exploding,
    ) as dbg:
        for step in range(num_steps):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                dbg.step = step

                if nan_step is not None and step >= nan_step and not nan_injected:
                    batch_x[0, :, :, :] = float("nan")
                    nan_injected = True

                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                dbg.record_loss(loss.item())
                optimizer.step()
                break
    return dbg


def analyze_results(dbg):
    """Extract all causal analysis results from a trained dbg instance."""
    return {
        "hypotheses": dbg.explain_failure("vanishing_gradients")
        + dbg.explain_failure("exploding_gradients"),
        "opt_hypotheses": dbg.explain_failure("optimizer_instability"),
        "data_hypotheses": dbg.explain_failure("data_anomaly"),
        "couplings": dbg.detect_coupled_failures(),
        "events": dbg.events,
        "mermaid": dbg.export_mermaid_causal_graph(),
    }


def scenario_vanishing_gradients(num_steps=30):
    """ResNet-18 with Tanh + small init + aggressive vanishing threshold."""
    model = create_resnet18(activation="tanh")

    def _small_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.005)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(_small_init)

    loader = _make_loader(100, 32)
    return train_resnet(
        model, loader, num_steps=num_steps, lr=1e-5, threshold_vanishing=1.0
    )


def scenario_exploding_gradients(num_steps=20):
    """ResNet-18 with very high LR."""
    model = create_resnet18(activation="relu")
    loader = _make_loader(100, 32)
    return train_resnet(
        model, loader, num_steps=num_steps, lr=10.0, threshold_exploding=50.0
    )


def scenario_data_anomaly(num_steps=20):
    """ResNet-18 with NaN injection at step 5."""
    model = create_resnet18(activation="relu")
    loader = _make_loader(100, 32)
    return train_resnet(model, loader, num_steps=num_steps, lr=0.01, nan_step=5)


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] ResNet-18 failure scenarios\n")

    for name, fn in [
        ("Vanishing Gradients (Tanh + small LR)", scenario_vanishing_gradients),
        ("Exploding Gradients (high LR)", scenario_exploding_gradients),
        ("Data Anomaly (NaN injection)", scenario_data_anomaly),
    ]:
        dbg = fn(num_steps=30)
        results = analyze_results(dbg)
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {name}")
        print(f"{'=' * 60}")
        print(f"Events: {len(results['events'])}")
        for label, hyps in [
            ("Causal hypotheses", results["hypotheses"]),
            ("Optimizer hypotheses", results["opt_hypotheses"]),
            ("Data anomaly", results["data_hypotheses"]),
        ]:
            if hyps:
                print(f"{label}:")
                for h in hyps:
                    print(f"  [{h.confidence:.2f}] {h.description}")
        if results["couplings"]:
            print("Coupled failures:")
            for c in results["couplings"]:
                d = c.get("step_difference", 0)
                print(
                    f"  {c['trigger']} -> {c['consequence']} (d={d}, {c['confidence']:.2f})"
                )

    print("\n[DONE] ResNet-18 scenarios complete.")


if __name__ == "__main__":
    main()
