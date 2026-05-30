## What it does

Your model's loss spikes to NaN or vanishes to zero. Instead of adding print statements and Googling for 2 hours, NeuralDBG hooks into your training loop and tells you:

1. **What** failed (vanishing gradients, exploding gradients, data anomalies)
2. **Where** (exact layer: "Tanh_3", not "somewhere in the middle")
3. **When** (exact step: "step 2", not "somewhere early")

```python
from neuraldbg import NeuralDbg

with NeuralDbg(model) as dbg:
    for step, (x, y) in dataloader:
        loss = train_step(model, x, y)
        dbg.record_loss(loss.item())

# After failure - get answers
hypotheses = dbg.explain_failure()
for h in hypotheses:
    print(h.description)  # "Gradient vanishing in layer Tanh_3 at step 2"
```

## Benchmark results

| Scenario | Detection | Localization | Step Accuracy |
|----------|-----------|-------------|---------------|
| Healthy training | 1.0 | 1.0 | 1.0 |
| Vanishing gradients | 1.0 | 1.0 | 1.0 |
| Exploding gradients | 1.0 | 1.0 | 1.0 |

**Overall: 100%**

## What is different from TensorBoard / W&B

| | TensorBoard | NeuralDBG |
|---|---|---|
| Shows | Metrics over time | **Why** it failed |
| Diagnosis | Manual | **Automated hypotheses** |
| Setup | Separate dashboard | **One line of code** |
| Privacy | Cloud | **100% local** |

## PyTorch compatibility

Tested against **7 PyTorch versions** (2.0.1 to 2.6.0) on Python 3.11/3.12 in CI.

## Links

- **GitHub**: [LambdaSection/NeuralDBG](https://github.com/LambdaSection/NeuralDBG)
- **PyPI**: `pip install neuraldbg`
- **Colab**: [Quickstart notebook](https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb)
- **Docs**: [Full tutorial](https://github.com/LambdaSection/NeuralDBG/blob/main/docs/tutorials/debug_pytorch_30s.md)

Feedback welcome - especially from anyone who has debugged vanishing/exploding gradients manually.
