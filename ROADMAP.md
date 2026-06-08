# NeuralSuite

> The complete toolkit for diagnosing and fixing deep learning training failures.

## What is NeuralSuite?

NeuralSuite is a three-part system that catches training problems before they waste your GPU hours:

| Component | What it does | Install |
|-----------|-------------|---------|
| **NeuralDBG** | Causal diagnostic engine — hooks into PyTorch, captures gradient/activation events, detects root causes | `pip install neuraldbg` |
| **Neural-Agent** | Auto-corrector — diagnoses failures and applies source-level fixes to training scripts | `pip install neural-agent` |
| **Aquarium** | Visualizer — interactive causal tree viewer for NeuralDBG exports | Desktop app (Tauri) |

## Why NeuralSuite?

Existing tools (W&B, MLflow, TensorBoard) are **dashboards** — they show you what happened. NeuralSuite tells you **why** it happened and **how to fix it**.

```
Training fails
    │
    ▼
NeuralDBG: "Vanishing gradients in layer_3, caused by Sigmoid saturation"
    │
    ▼
Neural-Agent: "Swap Sigmoid → LeakyReLU, increase LR by 2x"
    │
    ▼
Training runs successfully
```

## Quick Start

```python
from neuraldbg import NeuralDbg
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 64), nn.Sigmoid(), nn.Linear(64, 2))

with NeuralDbg(model) as dbg:
    for step in range(100):
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# See what went wrong
print(dbg.explain_failure())
```

## Roadmap

### v1.3.2 — Bug-Solver MVP (June 2026)
- [x] Composite-module hook support
- [x] Silent-loss and zero-leaf warnings
- [x] MHA fully-masked-row remediation rule
- [x] Neural-Agent: end-to-end diagnose -> fix -> validate pipeline
- [x] 300+ tests passing
- [x] Public benchmark: 4/4 scenarios at 1.0 accuracy
- [x] Tool comparison v2: NeuralDBG vs W&B vs MLflow vs TensorBoard
- [ ] First upstream PR submitted

### v1.4.5 — Catalog Expansion (July-August 2026)
- [ ] 10 real bugs cataloged (MHA, GNN, LSTM, GAN, diffusion, transformers, RL)
- [ ] Reproducible public benchmark on 5+ real scenarios
- [ ] Comparison vs Captum (explainability)
- [ ] 3+ upstream PRs submitted (at least 1 merged)
- [ ] Detection accuracy >= 0.90

### v1.5.0 — Obligation (August-September 2026)
- [ ] 20+ bugs cataloged, 10+ post-mortems published
- [ ] Versioned benchmark with CI regression gates
- [ ] Neural-Agent autonomous: closed loop on ResNet + Transformer + GAN
- [ ] 1+ upstream PR merged (external validation)
- [ ] Research paper draft on causal ML diagnostics

## Benchmark Results (v1.3.2)

4 scenarios, healthy excluded from averages:

| Tool | Detection (loss-only) | Detection (+grad norms) | Localization |
|------|:---------------------:|:-----------------------:|:------------:|
| **NeuralDBG** | **1.00** | **1.00** | **1.00** |
| W&B | 0.33 | 0.67 | 0.00 |
| MLflow | 0.33 | 0.67 | 0.00 |
| TensorBoard | 0.33 | 0.67 | 0.00 |

Key findings:
- External tools can detect anomalies (NaN loss, loss spikes) but cannot localize the failing layer
- With gradient norm logging, external tools detect MHA NaN gradients (0.33 -> 0.67)
- NeuralDBG is the only tool that names the layer causing the failure
- Benchmark is reproducible: `python -m benchmark_public.run`

## Install

```bash
pip install neuraldbg
```

## License

MIT
