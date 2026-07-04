# POST-006 — Silent NaN Swallowing: When svdvals Lies About Your Data

> **BUG-006** | **Source**: pytorch/pytorch#187759 | **PR**: #188053 (1ère review humaine — albanD)
> **Date**: 2026-07-04 | **Detection**: DeepMLP | **Gap**: +2 | **Causal Chain**: ✅

---

## 1. The Bug

**What**: `torch.linalg.svdvals()` silently returns finite singular values for matrices containing NaN. `torch.linalg.svd()` correctly propagates NaN. The two should be consistent — they're not.

**Upstream**: [pytorch#187759](https://github.com/pytorch/pytorch/issues/187759) — OPEN.

This is a **silent correctness bug**: NaN goes in, finite numbers come out. No error, no warning, no NaN. The computation appears to succeed while being mathematically wrong.

## 2. Reproduction

```python
import torch
A = torch.tensor([[1., 2., 3.],
                  [4., float('nan'), 6.],
                  [7., 8., 9.]])
result = torch.linalg.svdvals(A)
print(result)  # tensor([16.8481, 1.0684, 0.0000]) — WRONG
# Should be: NaN (because input has NaN)
```

Compare with `svd()`:
```python
U, S, Vh = torch.linalg.svd(A)
print(S)  # tensor([nan, nan, nan]) — CORRECT
```

## 3. Why It Matters

`svdvals` is used in production for:
- **Spectral normalization** (GAN training)
- **Condition number estimation** (numerical stability checks)
- **Low-rank approximations** (model compression)

When a NaN sneaks into the input (from a dead neuron, corrupted data, or buggy normalization), `svdvals` silently returns values that look reasonable. The training continues with corrupted gradients — **invisible to standard monitoring**.

## 4. NeuralDBG Diagnosis (DeepMLP)

| Phase | Anomalies | Events |
|-------|-----------|--------|
| Healthy | 2 | 48 |
| Bug (NaN input) | 4 | 50 |
| After fix | 0 | 46 |

**Causal Chain**:
```
data_anomaly[distribution_shift] → activation_regime_shift[saturated] → gradient_health_transition[healthy]
```

The chain shows: NaN in data → saturation in activations → gradient appears healthy but is silently wrong.

## 5. PR Status

- **#188053** submitted with minimal test (< 30 lines)
- **albanD** (PyTorch collaborator) reviewed — process feedback given
- Issue #187759 has actionable request posted
- Test validates NaN propagation consistency

## 6. The Fix

`svdvals` should propagate NaN when the input matrix contains NaN — same behavior as `svd()`:
```python
# Expected: svdvals(nan_matrix) → contains NaN
assert torch.isnan(torch.linalg.svdvals(A)).any()
```

---

*Detected by [NeuralDBG](https://github.com/LambdaSection/NeuralDBG). See all [post-mortems](index.html).*
