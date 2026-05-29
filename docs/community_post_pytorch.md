# Showcase: Debugging Exploding/Vanishing Gradients and NaNs in PyTorch with Causal Event Tracing (Introducing NeuralDBG)

Hey everyone!

If you train deep neural networks in PyTorch, you’ve probably spent hours dealing with training instability:
*   A loss suddenly spikes to `NaN` (because of an overflow or a bad softmax division).
*   Gradients disappear completely in the middle of training (vanishing gradients).
*   Activations saturate or ReLUs die across multiple layers.

Standard dashboards (TensorBoard, Weights & Biases, MLflow) are great for *metric logging*, but they are passive. They show you *that* something broke, but finding *why* and *where* it originated requires manually adding print statements, logging hooks, and tracing variables back in time.

To solve this, we built **NeuralDBG**—an open-source Python library that installs lightweight hooks on your leaf modules to perform **automated causal root-cause analysis** of training failures.

👉 **GitHub**: [https://github.com/LambdaSection/NeuralDBG](https://github.com/LambdaSection/NeuralDBG)
👉 **PyPI**: `pip install neuraldbg`

---

## How it Works under the Hood

NeuralDBG hooks into PyTorch's autograd engine and forward/backward passes:
1. **Semantic Event Capture**: During training, forward/backward hooks monitor activations, inputs, and gradient norms. They capture transition events (e.g. `DATA_ANOMALY` for NaNs/Infs, `ACTIVATION_REGIME_SHIFT` for dead/saturated activations, or `GRADIENT_HEALTH_TRANSITION`).
2. **Abductive Causal Reasoning**: When a failure is detected (like a loss divergence or gradient collapse), NeuralDBG traces back the dependency graph of events across steps and layers to isolate the first layer that failed and rank the causal hypotheses.
3. **Preventing OOM (TensorDiskCache)**: Storing full activation tensors in RAM/VRAM to inspect crashes usually causes Out-Of-Memory (OOM) errors. NeuralDBG solves this by only caching tensor statistics natively and dumping full anomalous tensors to disk using a lightweight `TensorDiskCache` *only* during state transitions.

---

## 30-Second Quickstart Demo (Colab Ready)

Here is a simple example of a deep MLP sabotaged with extremely small weights to force vanishing gradients. NeuralDBG detects it and points you to the exact source.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from neuraldbg import NeuralDbg

# 1. Create a deep network and sabotage the weights to force vanishing gradients
layers = []
input_dim = 20
for _ in range(8):
    layers.append(nn.Linear(input_dim, 20))
    layers.append(nn.ReLU())
    input_dim = 20
layers.append(nn.Linear(20, 1))
model = nn.Sequential(*layers)

with torch.no_grad():
    for param in model.parameters():
        param.fill_(1e-5)  # Tiny weights -> forces vanishing gradients

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 2. Wrap training with NeuralDbg
print("Training a sabotaged model with NeuralDBG...")
with NeuralDbg(model) as dbg:
    for step in range(5):
        optimizer.zero_grad()
        dbg.step = step
        
        x = torch.randn(8, 20)
        y = torch.randn(8, 1)
        
        loss = criterion(model(x), y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# 3. Request the causal explanation
print("\n--- NeuralDBG Causal Analysis ---")
hypotheses = dbg.explain_failure()
for i, h in enumerate(hypotheses, 1):
    print(f"\nHypothesis #{i} [Confidence: {h.confidence:.0%}]")
    print(f"  Description : {h.description}")
    print(f"  Causal Chain: {' -> '.join(h.causal_chain)}")
```

### Example Causal Output:
```text
Hypothesis #1 [Confidence: 90%]
  Description : gradient_vanishing detected at Linear_0 (step 0)
  Causal Chain: Linear_0@0 -> Linear_2@0 -> Linear_4@0 -> Linear_6@0
```

---

## Visualizing with Mermaid Graphs & Aquarium

NeuralDBG can export a Mermaid diagram representing the causal flow:
```python
print(dbg.export_mermaid_causal_graph())
```
It also exports full JSON diagnostic packages:
```python
dbg.export_aquarium_package("report.json")
```
Which can be rendered in our local Tauri-based visualizer (**Aquarium**) to inspect activation distributions, resource utilization metrics (CPU RAM and GPU VRAM spikes), and gradient flow history visually.

---

## Feedback & Open Technical Questions

We are currently looking for feedback from the community, especially regarding:
1. **Handling `torch.compile`**: We've added compatibility guards, but hooks registered on leaf modules behave differently after compilation. How do you handle fine-grained module tracking inside compiled graphs?
2. **Distributed Training (`DistributedDataParallel`)**: We currently emit warnings when DDP is wrapped directly and recommend wrapping the inner module. If you train on multi-GPU setups, what features would be most useful for synchronization?

Check out the code, try it on your models, and let us know what you think!

*NeuralDBG is licensed under the MIT License.*
