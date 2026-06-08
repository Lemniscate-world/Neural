# [D] NeuralDBG — Causal root-cause analysis when PyTorch training fails

Hello PyTorch community,

I've been building **NeuralDBG** — an open-source causal inference engine for PyTorch training loops. When your loss goes to NaN or gradients vanish, it pinpoints the exact layer and step where the failure started.

## The problem it solves

When training breaks, the debugging workflow is:
1. Stare at the loss curve
2. Add `print(grad.norm())` in the loop
3. Guess which layer is responsible
4. Try random fixes

NeuralDBG automates steps 2-4 by installing hooks that capture layer-level gradient and activation statistics, then generates ranked causal hypotheses.

## Quick start

```python
from neuraldbg import NeuralDbg

with NeuralDbg(model) as dbg:
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        dbg.step = step
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# After a failure, query explanations
hypotheses = dbg.explain_failure()
for h in hypotheses:
    print(f"{h.layer}: {h.description} (confidence: {h.confidence:.2f})")
```

Output when gradients vanish:
```
Tanh_3: Gradient vanishing detected at step 2 (confidence: 1.00)
  Transition: grad_norm dropped from 0.45 to 0.00 between steps 1 and 2
```

## How it differs from TensorBoard / W&B

| | TensorBoard / W&B | NeuralDBG |
|---|---|---|
| Shows **when** metrics moved | Yes | Yes |
| Shows **which layer** caused it | No | Yes |
| Generates ranked hypotheses | No | Yes |
| Detects gradient NaN in specific layers | No | Yes |

## What it does under the hood

- **Backward hooks** track per-layer gradient norms and detect health transitions
- **Forward hooks** capture activation statistics (sparsity, dead neurons, saturation)
- **Event compression** identifies first occurrences and propagation patterns
- **Abductive reasoning** generates hypotheses ranked by confidence
- **Composite module support** for nn.MultiheadAttention, etc. (v1.3.2)

## Public benchmark

5 scenarios, reproducible: `python -m benchmark_public.run`

| Scenario | Detection | Localization | Accuracy |
|----------|-----------|-------------|----------|
| Healthy training | 1.0 | 1.0 | 1.0 |
| Vanishing gradients | 1.0 | 1.0 | 1.0 |
| Exploding gradients | 1.0 | 1.0 | 1.0 |
| MHA fully-masked row (BUG-001) | 1.0 | 1.0 | 1.0 |
| NaN loss from layer injection | 1.0 | 1.0 | 1.0 |

We also ran a real comparison against W&B, MLflow, and TensorBoard on the same scenarios (loss-only mode):

| Tool | Detection | Localization |
|------|-----------|-------------|
| **NeuralDBG** | **1.00** | **1.00** |
| W&B | 0.50 | 0.00 |
| MLflow | 0.50 | 0.00 |
| TensorBoard | 0.50 | 0.00 |

## Real bugs found

We've used NeuralDBG to diagnose real upstream bugs:
- **BUG-001**: pytorch#41508 — NaN gradients in nn.MultiheadAttention with masked rows
- **BUG-002**: pytorch#176793 — NaN gradients in varlen_attn with padding
- **BUG-003**: pytorch#177116 — MPS backward pass produces 1000x-100000x wrong gradients

## Links

- GitHub: https://github.com/LambdaSection/NeuralDBG
- PyPI: `pip install neuraldbg`
- Blog: https://lambdasection.github.io/NeuralDBG/blog/

## Compatibility

- Python 3.9+ / PyTorch 2.0+
- Works with nn.DataParallel
- CPU and CUDA

Feedback welcome — especially if you've ever stared at a loss curve wondering "why did this die?"
