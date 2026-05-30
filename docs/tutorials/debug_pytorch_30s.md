# Debug Your PyTorch Training in 30 Seconds — NeuralDBG Tutorial

**TL;DR**: Your model's loss suddenly spikes or vanishes and you don't know why. NeuralDBG hooks into your training loop, detects the failure type, localizes the exact layer, and generates a ranked hypothesis — all automatically. Here's how.

---

## The Problem

You're training a neural network. Everything looks fine, then suddenly:

```
Step 0: loss=0.6931
Step 1: loss=0.6812
Step 2: loss=nan          ← 💀
```

You add print statements. You check gradients. You reduce the learning rate. You Google "loss nan pytorch". You waste 2 hours.

**What if your training loop could tell you exactly what went wrong?**

---

## The 30-Second Fix

```python
import torch
import torch.nn as nn
from neuraldbg import NeuralDbg

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.Tanh(),
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128, 10),
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))

# Wrap your loop with NeuralDbg
with NeuralDbg(model) as dbg:
    for step in range(20):
        optimizer.zero_grad()
        dbg.step = step

        # Simulate a vanishing gradient at step 3
        if step == 3:
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(0.0)  # 💥 Corrupt weights

        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# Ask NeuralDBG what happened
hypotheses = dbg.explain_failure()
for h in hypotheses:
    print(f"→ {h.description}")
    print(f"  Confidence: {h.confidence:.2f}")
    print(f"  Evidence: {[(e.layer_name, e.step) for e in h.evidence]}")
```

**Output:**
```
→ Gradient vanishing originated in layer 'Tanh_3' at step 2
  Confidence: 1.00
  Evidence: [('Tanh_3', 2)]

→ Root cause candidate: gradient vanishing originated in 'Linear_0' at step 2
  Confidence: 0.95
  Evidence: [('Linear_0', 2)]
```

NeuralDBG detected the vanishing gradient, localized it to `Tanh_3` at step 2, and ranked the hypotheses by confidence. **No manual inspection needed.**

---

## What NeuralDBG Actually Does

NeuralDBG installs PyTorch hooks (forward + backward) on every module in your model. At each training step, it extracts **semantic events** — not raw tensors, but meaningful transitions:

| Event Type | What It Detects |
|------------|-----------------|
| `gradient_health_transition` | Gradient norm drops below threshold (vanishing) or explodes |
| `activation_regime_shift` | Activations become saturated, dead, or NaN |
| `optimizer_instability` | Loss spikes, plateaus, or divergence |
| `data_anomaly` | NaN/Inf in inputs, distribution shifts |

These events are compressed into **causal hypotheses** using first-occurrence tracking and propagation analysis.

### The Key Insight

Most debugging tools show you **metrics over time** (loss curves, gradient histograms). NeuralDBG shows you **causes** — which layer failed first, when it happened, and what propagated from there.

---

## Real Example: Vanishing Gradients in a Deep Network

Here's a 10-layer network with Tanh activations. After step 2, all weights are zeroed (simulating catastrophic weight corruption).

![Vanishing Gradient Detection](../outputs/vanishing_gradient_demo.gif)

The heatmap shows:
- **Green** = healthy gradients (norm > 1e-4)
- **Red + X** = vanishing gradients (norm < 1e-4)
- **Dashed line** = injection point (step 2)

NeuralDBG transitions from "healthy" to "vanishing" at exactly the injection step, across all layers simultaneously.

---

## Why Not TensorBoard / W&B?

| | TensorBoard / W&B | NeuralDBG |
|---|---|---|
| **Shows** | Loss curves, gradient histograms | **Why** the loss spiked |
| **Diagnosis** | Manual inspection | **Automatic** causal hypotheses |
| **Action** | You guess the fix | **Suggests root causes** |
| **Setup** | Separate dashboard | **One line** in your loop |
| **Privacy** | Data sent to cloud | **100% local** |

> "TensorBoard tells you *when* it failed. NeuralDBG tells you *why*."

---

## Installation

```bash
pip install neuraldbg
```

Requires Python 3.9+ and PyTorch 2.0+. Tested against PyTorch 2.0 → 2.6 in CI.

---

## What's Next

NeuralDBG is the **diagnostic engine**. The next pieces:

1. **Aquarium** — Visual IDE for exploring causal graphs interactively
2. **Neural-Agent** — Auto-remediation: the agent reads NeuralDBG's output and fixes your training loop

The JSON export format from NeuralDBG is designed to be consumed by other tools — it's a **protocol**, not just a library.

---

## Links

- **GitHub**: [github.com/LambdaSection/NeuralDBG](https://github.com/LambdaSection/NeuralDBG)
- **PyPI**: [pypi.org/project/neuraldbg](https://pypi.org/project/neuraldbg/)
- **Quickstart Colab**: [Open in Colab](https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb)
- **Benchmark**: 100% detection, 100% localization on 3 synthetic failure scenarios

---

*If you've ever stared at a loss curve wondering "why did this die?", give NeuralDBG a try. It takes 30 seconds to add and might save you hours of debugging.*
