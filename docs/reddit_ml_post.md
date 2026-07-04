# [P] NeuralDBG — Causal Debugging for PyTorch Training Failures

**TL;DR**: We built a tool that hooks into PyTorch's autograd and tells you WHY your training failed (root cause → symptom chain), not just WHAT happened. 89% detection rate with 0% false positives on ResNet + Transformer architectures. Looking for feedback.

---

## The Problem We All Face

Training crashes with NaN loss at step 3,247. You spend the next 2 hours checking:
- Learning rate too high? (maybe)
- Bad data batch? (probably not)
- Gradient explosion in layer 14? (how would you know?)
- Some weird optimizer state interaction? (good luck)

Existing tools show you dashboards. They don't tell you WHY.

## What NeuralDBG Does

Drop-in context manager, zero config:

```python
from neuraldbg import NeuralDbg

with NeuralDbg(model) as dbg:
    for step in range(steps):
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())

# Get causal chains — ranked by severity
chains = dbg.explain_causal()
# data_anomaly[distribution_shift] → gradient[exploding] → optimizer_instability[diverging]
```

## Results (validated July 2026)

- **Mini ResNet (CNN)**: 4/5 bugs caught, 0 false positives
- **Mini Transformer (attention)**: 5/5 bugs caught, 0 false positives  
- **DeepMLP (12-layer residual)**: 7/7 bugs caught, 0 false positives
- **Combined: 89% detection, 0% FP**

Reproduce: `pip install neuraldbg && python validate_real_architectures.py`

## Comparison

| | Causal Chain | Layer Diagnosis | Non-invasive | Open Source |
|---|:---:|:---:|:---:|:---:|
| NeuralDBG | ✅ | ✅ | ✅ | MIT |
| W&B | ❌ | ❌ | ✅ | ❌ |
| Captum | ❌ | ✅ | ❌ | BSD |
| TensorBoard | ❌ | ❌ | ✅ | Apache |

## We Want Your Feedback

1. What training bugs waste YOUR time?
2. Would you use a hook-based diagnostic in production training?
3. What's missing from the current detection (gradients, activations, optimizer state)?

**GitHub**: [github.com/LambdaSection/NeuralDBG](https://github.com/LambdaSection/NeuralDBG)  
**Install**: `pip install neuraldbg`  
**Docs**: [lambdasection.github.io/NeuralDBG](https://lambdasection.github.io/NeuralDBG)

---

*We've also submitted 4 upstream PRs to PyTorch fixing bugs found during development. Open to collaborations!*
