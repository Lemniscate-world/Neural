# POST-007 — The Compiler Lies: torch.compile Silent Gradient Corruption

> **BUG-007** | **Source**: pytorch/pytorch#186799 | **Reported by**: @ezyang  
> **Date**: 2026-07-04 | **Detection**: DeepMLP | **Gap**: +24

---

## 1. The Bug

**What**: `torch.compile` silently produces wrong gradients for `torch.atan2` when the denominator approaches zero. The eager mode computes correct gradients. The compiled mode computes different (wrong) gradients — with no warning.

**Upstream**: [pytorch#186799](https://github.com/pytorch/pytorch/issues/186799) — OPEN, reported by @ezyang (PyTorch core developer).

This is especially dangerous because `torch.compile` is marketed as a drop-in optimization — users expect identical results, not silently corrupted gradients.

## 2. Reproduction

```python
import torch

@torch.compile
def f(x, y):
    return torch.atan2(x, y.abs() + 1e-8)

x = torch.randn(4, requires_grad=True)
y = torch.randn(4, requires_grad=True)

# Eager: correct gradients
out_eager = torch.atan2(x, y.abs() + 1e-8)
out_eager.sum().backward()
g_eager = x.grad.clone()

# Compiled: silently wrong gradients
x.grad = None
out_compiled = f(x, y)
out_compiled.sum().backward()
g_compiled = x.grad

# g_eager != g_compiled — silent correctness violation!
```

## 3. NeuralDBG Diagnosis (DeepMLP)

| Phase | Anomalies | Events |
|-------|-----------|--------|
| Healthy | 0 | 46 |
| Bug (zero denominator) | 24 | 70 |

**Gap: +24** — strong signal despite the bug being a compiler-level issue.

## 4. Why It Matters

- **Reported by a PyTorch core dev** — high credibility bug
- **Compiler bugs are the hardest to find** — silent, no NaN, no crash
- **Affects everyone using torch.compile** — which is increasingly the default

---

*Detected by [NeuralDBG](https://github.com/LambdaSection/NeuralDBG). See all [post-mortems](index.html).*
