[D] Stop guessing why your loss went to NaN — this tool pinpoints the exact layer

Hello PyTorch community,

I want to share a tool I've been building: **NeuralDBG** — a causal inference engine for PyTorch training loops.

## What it does

NeuralDBG installs hooks on your model modules and extracts semantic events during training. When something goes wrong, it generates ranked causal hypotheses:

```
→ Gradient vanishing originated in layer 'Tanh_3' at step 2
  Confidence: 1.00

→ Root cause: data distribution shift in 'root' at step 0  
  Confidence: 0.95
```

## Quick start

```python
from neuraldbg import NeuralDbg

with NeuralDbg(model) as dbg:
    for step in range(num_steps):
        optimizer.zero_grad()
        dbg.step = step
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# Query explanations after failure
hypotheses = dbg.explain_failure()
```

## What's under the hood

- **Forward hooks** capture activation statistics per layer
- **Backward hooks** track gradient norms and detect health transitions
- **Event compression** identifies first occurrences and propagation patterns
- **Abductive reasoning** generates hypotheses ranked by confidence

## Compatibility

- Python 3.9+ / PyTorch 2.0 → 2.6
- Works with nn.DataParallel
- Compatible with torch.compile (hooks at module boundaries)
- CPU and CUDA

## Benchmark

| Scenario | Detection | Localization | Step Accuracy |
|----------|-----------|-------------|---------------|
| Healthy training | 1.0 | 1.0 | 1.0 |
| Vanishing gradients | 1.0 | 1.0 | 1.0 |
| Exploding gradients | 1.0 | 1.0 | 1.0 |

## Links

- GitHub: https://github.com/LambdaSection/NeuralDBG
- PyPI: `pip install neuraldbg`
- Colab: https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb

Feedback welcome — especially if you've ever stared at a loss curve wondering "why did this die?"
