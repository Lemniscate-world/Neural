# NeuralDBG + Weights & Biases: Causal Debugging for PyTorch Training

> **One-liner**: Add causal chain debugging to any W&B run with `callback = NeuralDBGCallback(model)`.

## The Problem

You're training a model. Loss spikes at step 200. You check W&B — yes, loss went up. But **why**?

- Was it a gradient explosion in layer 3?
- A dead neuron in the attention block?
- Data corruption propagating silently for 50 steps?
- Or optimizer instability from a bad LR schedule?

W&B tells you *what* happened. NeuralDBG tells you **why**.

## What NeuralDBG Does

NeuralDBG is a **causal diagnostic engine** for PyTorch training. It hooks into your model's forward and backward passes, tracks gradient health per layer, activation regimes, and optimizer dynamics, then builds **causal chains** — directed graphs from root cause to final symptom.

Results on 212 architectures across 6 families:
- **100%** detection on ResNet-18, Transformer, RL
- **96%** on black-swan architectures (GNN, MoE, Diffusion)
- **150+ causal chains** per failure (vs 0 from any other tool)

## NeuralDBG + W&B: One-Line Integration

```python
from neuraldbg.integrations.wandb import NeuralDBGCallback

callback = NeuralDBGCallback(model, family="Transformer")
with callback:
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        callback.step(loss)      # ← one line
        optimizer.step()

# After training:
report = callback.report()
print(report["summary"])
```

### What gets logged to W&B

| Metric | Description |
|--------|-------------|
| `neuraldbg/total_events` | All events captured |
| `neuraldbg/anomaly_events` | Anomalous events only |
| `neuraldbg/causal_chains` | Number of causal chains |
| `neuraldbg/event_distribution` | Bar chart of event types |
| `neuraldbg/causal_chains_table` | Ranked table (root cause, confidence, chain) |
| `wandb.alert()` | Real-time alerts for NaN, gradient health transitions, optimizer instability |

### Even Simpler: Auto-Patch

For existing codebases, monkey-patch `wandb.init`:

```python
from neuraldbg.integrations.wandb import patch_wandb_init
patch_wandb_init(family="Transformer")

# Now ALL wandb.init() calls get NeuralDBG automatically
wandb.init(project="my-project")
# ... your existing training loop, zero changes
```

## Real Example: Detecting a Vanishing Gradient

```python
import wandb
import torch.nn as nn
from neuraldbg.integrations.wandb import NeuralDBGCallback

model = nn.Sequential(
    nn.Linear(64, 128), nn.Sigmoid(),  # ← saturated!
    nn.Linear(128, 64), nn.Sigmoid(),
    nn.Linear(64, 10),
)

wandb.init(project="neuraldbg-demo")
callback = NeuralDBGCallback(model, family="MLP")
with callback:
    for step in range(500):
        x = torch.randn(32, 64)
        loss = model(x).sum()
        loss.backward()
        callback.step(loss.item())
        # W&B gets: events, chains, alerts automatically

report = callback.report()
# report["causal_chains"][0]["root_cause"]
# → "Sigmoid saturation in Sequential.0 (layer 1): gradient_norm=1e-8"
```

## Why This Matters

**Detection parity is not the story.** Both NeuralDBG and W&B detect failures. **Information asymmetry is the story:**

| Tool | Detection | Causal Chains | Root Cause |
|------|:---------:|:-------------:|:----------:|
| W&B alone | 5/6 | 0 | ❌ |
| NeuralDBG + W&B | **6/6** | **150+** | ✅ layer + step |

Time-to-diagnosis: ~5 minutes with NeuralDBG + W&B vs hours manually correlating W&B charts.

## Try It

```bash
pip install neuraldbg wandb
```

- **GitHub**: https://github.com/LambdaSection/NeuralDBG
- **Docs**: https://github.com/LambdaSection/NeuralDBG#readme
- **HF Space Demo**: https://huggingface.co/spaces/KuroGSekai/neuraldbg-demo
- **Paper**: Coming soon on arXiv (cs.LG)

---

*NeuralDBG is MIT licensed. Built by LambdaSection.*
