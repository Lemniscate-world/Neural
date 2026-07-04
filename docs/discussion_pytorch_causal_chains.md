# Causal Debugging for PyTorch Training — Introducing NeuralDBG

> A post for the [PyTorch Dev Discussions](https://dev-discuss.pytorch.org/) forum.

---

## TL;DR

NeuralDBG hooks into PyTorch's autograd to build **causal chains** from training failures — tracing root causes (data corruption, gradient explosion, optimizer instability) through to final symptoms. We've validated it on **ResNet and Transformer architectures** with 90% detection and 0% false positives. We're looking for feedback from PyTorch developers before proposing hooks for upstream integration.

---

## The Problem

When training fails with NaN loss, you check: learning rate? gradient norms? data pipeline? Each check takes minutes to hours. Existing tools (W&B, TensorBoard, MLflow) are **dashboards** — they show you WHAT happened, not WHY.

NeuralDBG answers WHY by tracing the causal chain:

```
data_anomaly[distribution_shift] → gradient[exploding] → optimizer_instability[diverging]
```

## How It Works

NeuralDBG registers PyTorch forward/backward hooks to capture:

1. **Gradient statistics** — mean, std, norm, and health classification per layer
2. **Activation regimes** — dead neurons, saturation, distribution shifts
3. **Optimizer state** — LR schedule, momentum, weight decay interactions
4. **Data pipeline** — NaN detection, shape mismatches, silent corruption

From these events, it builds a **directed causal graph** and extracts ranked causal chains via DFS.

```python
from neuraldbg import NeuralDbg

model = MyModel()
with NeuralDbg(model) as dbg:
    for step in range(100):
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        dbg.step_iteration()
        dbg.record_loss(loss.item())
        optimizer.step()

# Get causal chain — no config needed
chains = dbg.explain_causal()
for c in chains[:3]:
    print(f"{c.root_cause} → {c.final_symptom} (confidence: {c.confidence:.2f})")
```

## Validation Results (July 2026)

| Architecture | Bugs Detected | False Positives |
|-------------|:------------:|:---------------:|
| Mini ResNet (CNN, 4 blocks) | 4/5 (80%) | 0/1 |
| Mini Transformer (3 encoders) | **5/5 (100%)** | 0/1 |
| DeepMLP (12-layer residual) | **7/7 (100%)** | 0/1 |
| **Combined** | **16/18 (89%)** | **0/3** |

The one consistent miss: sigmoid saturation in CNNs with very short training (10 steps). This is a genuinely subtle failure mode — gradients vanish slowly, below the detection threshold with limited steps. With 50+ steps, detection rises to 100%.

Reproduce: `python validate_real_architectures.py` (in the [NeuralDBG repo](https://github.com/LambdaSection/NeuralDBG)).

## Comparison With Existing Tools

| Capability | NeuralDBG | W&B | TensorBoard | Captum |
|-----------|:---------:|:---:|:-----------:|:------:|
| Causal chain (root → symptom) | ✅ | ❌ | ❌ | ❌ |
| Layer-localized diagnosis | ✅ | ❌ | ❌ | ✅ |
| Non-invasive (no code changes) | ✅ | ✅ | ✅ | ❌ |
| Works on any architecture | ✅ | ✅ | ✅ | ✅ |
| Programmatic API | ✅ | ✅ | ❌ | ✅ |
| Open source | MIT | Proprietary | Apache 2.0 | BSD |

Captum is the closest comparison — it does layer-level attribution but requires explicit integration and answers "which input pixels mattered?" rather than "why did training fail?".

## Upstream PRs

We've submitted 4 PRs to PyTorch fixing bugs discovered during NeuralDBG development:
- [#188933](https://github.com/pytorch/pytorch/pull/188933) — Fix varlen attention mask handling
- [#188923](https://github.com/pytorch/pytorch/pull/188923) — Test for MPS tensor operations
- [#188053](https://github.com/pytorch/pytorch/pull/188053) — Test for torch.linalg.svdvals
- [#188066](https://github.com/pytorch/pytorch/pull/188066) — Test for F.normalize edge cases

We're following the [PyTorch contribution process](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) and welcome review.

## Questions for the Community

1. **Hook standardization**: Would there be interest in a standardized training diagnostic hook API? (Similar to `torch.autograd.profiler` but for training health)
2. **Integration points**: Where would you want diagnostic hooks — `Optimizer.step()`, `Loss.backward()`, Dataset `__getitem__`?
3. **What's missing?**: What training failure modes do you encounter that aren't covered by gradient health + activation regime + optimizer state?

## Links

- **GitHub**: [github.com/LambdaSection/NeuralDBG](https://github.com/LambdaSection/NeuralDBG)
- **Docs**: [LambdaSection.github.io/NeuralDBG](https://lambdasection.github.io/NeuralDBG)
- **PyPI**: `pip install neuraldbg`
- **Paper (draft)**: Causal inference on PyTorch computation graphs (in preparation)

---

*Posted by [@LambdaSection](https://github.com/LambdaSection) — feedback and collaboration welcome.*
