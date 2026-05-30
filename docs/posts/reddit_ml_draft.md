[D] NeuralDBG: Causal inference engine that tells you WHY your PyTorch training failed (not just when)

Hey r/MachineLearning,

I built a tool that hooks into your PyTorch training loop and automatically generates **causal hypotheses** when training fails.

**The problem**: Your loss spikes to NaN, or gradients vanish, and you spend hours adding print statements, checking gradients, Googling "loss nan pytorch fix"... only to discover it was a bad learning rate or a corrupted weight at layer 4.

**What NeuralDBG does**:
- Detects failure type (vanishing/exploding gradients, data anomalies, optimizer instability)
- Localizes the exact layer where the failure originated
- Generates ranked hypotheses with confidence scores

```python
from neuraldbg import NeuralDbg

with NeuralDbg(model) as dbg:
    for step, (x, y) in dataloader:
        loss = train_step(model, x, y)
        dbg.record_loss(loss.item())

# After failure - ask what went wrong
for h in dbg.explain_failure():
    print(h.description)
    # → "Gradient vanishing originated in layer 'Tanh_3' at step 2"
    # → "Root cause candidate: data distribution shift in 'root' at step 0"
```

**Benchmark**: 100% detection, 100% localization, 100% step accuracy on 3 synthetic failure scenarios (healthy, vanishing, exploding).

**Key design decisions**:
- Extracts **semantic events** (transitions), not raw tensors — compact and meaningful
- **One line** to add to existing code (context manager wrapper)
- **100% local** — no cloud, no telemetry
- Tested against PyTorch 2.0 → 2.6 in CI

**Links**:
- GitHub: https://github.com/LambdaSection/NeuralDBG
- PyPI: `pip install neuraldbg`
- Colab quickstart: https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb

Would love feedback from anyone who has debugged vanishing/exploding gradients. Does this solve a real problem for you?
