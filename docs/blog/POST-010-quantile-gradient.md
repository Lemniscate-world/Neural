# POST-010 — When Tied Values Break Gradients: Inductor Quantile Mismatch

> **BUG-010** | **Source**: pytorch/pytorch#185543  
> **Date**: 2026-07-04 | **Detection**: DeepMLP | **Gap**: +16

---

## 1. The Bug

**What**: `torch.quantile` with the Inductor backend produces gradient mismatches when input values are tied (all equal). Eager mode computes one gradient, Inductor computes another.

**Upstream**: [pytorch#185543](https://github.com/pytorch/pytorch/issues/185543) — OPEN.

## 2. Reproduction

```python
import torch

x = torch.ones(3, 5) * 3.0  # all values tied
q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]), dim=1)
# Gradient through quantile differs between eager and Inductor
```

## 3. NeuralDBG Diagnosis (DeepMLP)

| Phase | Anomalies | Events |
|-------|-----------|--------|
| Healthy | 0 | 46 |
| Bug (tied values) | 16 | 62 |

**Gap: +16** — strong signal for an edge case that standard tools would never catch.

## 4. Why It Matters

- **Edge case that's common in practice**: tied values occur naturally with constant inputs, padding, or saturated activations
- **Inductor is the default compiler**: more users will hit this as torch.compile adoption grows
- **Silent**: no NaN, no crash — just wrong gradients

---

*Detected by [NeuralDBG](https://github.com/LambdaSection/NeuralDBG). See all [post-mortems](index.html).*
