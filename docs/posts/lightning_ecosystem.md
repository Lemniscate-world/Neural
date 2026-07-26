# NeuralDBG + PyTorch Lightning: Causal Debugging as a Callback

> **One-liner**: `trainer = pl.Trainer(callbacks=[NeuralDBGLightningCallback(family="CNN")])`

## The Problem

You're using Lightning because it removes boilerplate. But when training fails — NaN at step 300, loss plateau, gradient explosion — you're back to manual debugging:

- Adding print statements
- Bisecting checkpoints
- Correlating W&B charts by eye

Lightning handles the training loop. NeuralDBG handles the **why**.

## What NeuralDBG Does

NeuralDBG is a **causal diagnostic engine** that hooks into PyTorch's forward/backward passes and traces training failures to their root cause. It detects:

- **Gradient health**: vanishing, exploding, NaN per layer
- **Activation regime**: dead neurons, saturated sigmoids, collapsed attention
- **Optimizer dynamics**: instability, divergence
- **Data anomalies**: NaN/Inf propagation, silent corruption

It then builds **causal chains** — directed graphs from root cause → propagation → final symptom.

## NeuralDBG + Lightning: One Callback

```python
import pytorch_lightning as pl
from neuraldbg.integrations.lightning import NeuralDBGLightningCallback

trainer = pl.Trainer(
    callbacks=[
        NeuralDBGLightningCallback(family="CNN"),  # ← one line
    ]
)
trainer.fit(model, dataloader)
```

That's it. Events and causal chains are logged to whatever logger Lightning uses (W&B, TensorBoard, CSV).

### What gets logged

| Metric | Description |
|--------|-------------|
| `neuraldbg/total_events` | All events captured |
| `neuraldbg/anomaly_events` | Anomalous events only |
| `neuraldbg/causal_chains` | Number of causal chains traced |
| `neuraldbg/event_types` | Distinct event types |
| `neuraldbg/top_chain` | Top-ranked root cause text |

## How It Works Under the Hood

The callback hooks into Lightning's lifecycle:

1. `on_fit_start` — Initializes NeuralDBG on the LightningModule
2. `on_train_batch_end` — Records loss, runs diagnostics, logs summary every N steps
3. `on_fit_end` — Final summary and cleanup

No model wrapping, no training loop modification, no `loss.backward()` interception needed.

## Validated on Real Architectures

| Architecture | Bugs Detected | False Positives |
|-------------|:------------:|:---------------:|
| Mini ResNet (CNN, 4 blocks) | 4/5 (80%) | 0/1 |
| Mini Transformer (3 encoders) | **5/5 (100%)** | 0/1 |
| ResNet-18 (torchvision, 11M) | **6/6 (100%)** | 5 FP → reduced to 0 |
| **Total** | **15/16 (94%)** | **0/2** |

## Quick Start

```bash
pip install neuraldbg pytorch-lightning
```

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from neuraldbg.integrations.lightning import NeuralDBGLightningCallback

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        return loss

model = MyModel()
trainer = pl.Trainer(
    callbacks=[NeuralDBGLightningCallback(family="MLP", log_every_n_steps=50)],
    max_epochs=5,
)
trainer.fit(model, dataloader)

# After training: NeuralDBG has logged events + causal chains
# Check your logger (W&B/TensorBoard) for neuraldbg/* metrics
```

## Why This Matters

Unlike monitoring tools that just detect anomalies, NeuralDBG traces **why** they happened. On the honest benchmark (6 failure scenarios across identical architectures), NeuralDBG produced **150+ causal chains** while W&B/TensorBoard produced **zero**. Time-to-diagnosis drops from hours to ~5 minutes.

## Resources

- **GitHub**: https://github.com/LambdaSection/NeuralDBG
- **W&B Integration**: `neuraldbg.integrations.wandb.NeuralDBGCallback`
- **HF Space Demo**: https://huggingface.co/spaces/KuroGSekai/neuraldbg-demo
- **Paper**: Coming soon on arXiv (cs.LG)

---

*NeuralDBG is MIT licensed. Built by LambdaSection.*
