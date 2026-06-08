# ROADMAP.md — NeuralDBG

> Causal root cause analysis for deep learning training dynamics.

## Vision

NeuralDBG is the **diagnostic engine** for neural network training. It captures causal event chains (gradients, activations, loss) and automatically detects the root cause of training failures.

> A Python library alone isn't enough in the age of AI coding agents.
> NeuralDBG becomes the **structured protocol** (causal chains, event types, root causes) that an AI agent consumes to auto-diagnose and auto-repair training.

## Architecture

```
NeuralDBG (engine)  -->  Neural-Agent (auto-fixer)  -->  Aquarium (visualizer)
     |                         |                              |
  pip install neuraldbg    pip install neural-agent        Tauri IDE
```

## Roadmap

### v1.3.2 — Bug-Solver MVP (June 2026)
- [x] Composite-module hook support (FIX-001)
- [x] Silent-loss and zero-leaf warnings
- [x] MHA fully-masked-row remediation rule
- [x] Neural-Agent: end-to-end diagnose -> fix -> validate pipeline
- [x] 300+ tests passing
- [ ] Public benchmark with real bug scenarios
- [ ] First upstream PR submitted

### v1.4.5 — Catalog Expansion (July-August 2026)
- [ ] 10 real bugs cataloged (MHA, GNN, LSTM, GAN, diffusion, transformers, RL)
- [ ] Reproducible public benchmark on 5+ real scenarios
- [ ] Comparison vs W&B / MLflow / TensorBoard / Captum
- [ ] Neural-Agent published on PyPI
- [ ] 3+ upstream PRs submitted (at least 1 merged)
- [ ] Detection accuracy >= 0.90

### v1.5.0 — Obligation (August-September 2026)
- [ ] 20+ bugs cataloged, 10+ post-mortems published
- [ ] Versioned benchmark with CI regression gates
- [ ] Neural-Agent autonomous: closed loop on ResNet + Transformer + GAN
- [ ] 1+ upstream PR merged (external validation)
- [ ] Research paper draft on causal ML diagnostics

## Install

```bash
pip install neuraldbg
```

## Quick Start

```python
from neuraldbg import NeuralDbg
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 64), nn.Sigmoid(), nn.Linear(64, 2))

with NeuralDbg(model) as dbg:
    for step in range(10):
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

print(dbg.explain_failure())
```

## License

MIT
