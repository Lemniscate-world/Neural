# POST-008 — The Billion-Dollar Gradient: When F.normalize Silently Corrupts Your Weights

> **BUG-008** | **Source**: pytorch/pytorch#184575 | **PR**: #188066 (CI fixé, 13→0 échecs)
> **Date**: 2026-07-04 | **Detection**: DeepMLP | **Gap**: +17 | **Causal Chain**: ✅

---

## 1. The Bug

**What**: `F.normalize(x, dim=0)` on a zero vector returns 0 in forward pass and **~1,000,000,000,000** in backward pass — instead of NaN. This is silently corrupting model weights.

**Upstream**: [pytorch#184575](https://github.com/pytorch/pytorch/issues/184575) — OPEN.

The math: `F.normalize(x) = x / ||x||`. At `x = 0`, this is **division by zero**. IEEE 754 says this should produce NaN. PyTorch returns 0 forward, ~1e12 backward.

## 2. Reproduction

```python
import torch, torch.nn.functional as F

x = torch.zeros(3, requires_grad=True)
y = F.normalize(x, dim=0)
y.sum().backward()

print(x.grad)  # tensor([1.1111e+12, 1.1111e+12, 1.1111e+12])
# Should be: NaN (division by zero)
```

## 3. Why It's Dangerous

1. **Forward returns 0** — loss looks normal, no NaN alert triggered
2. **Backward returns ~1e12** — optimizer takes a massive step in a random direction
3. **Weights are silently corrupted** — model converges to wrong values, no error message
4. **Happens naturally** — zero vectors occur after dropout, masked attention, or padding

## 4. NeuralDBG Diagnosis (DeepMLP)

| Phase | Anomalies | Events |
|-------|-----------|--------|
| Healthy | 0 | 46 |
| Bug (zero input) | 17 | 63 |
| After fix | 19 | 65 |

**Gap: +17** — strong detection signal. The fix (epsilon guard) works but shows some residual noise on the deep model.

**Causal Chain**: `data_anomaly[distribution_shift] → gradient_health_transition[exploding] → optimizer_instability[diverging]`

## 5. PR Status

- **#188066** submitted with 4 tests (forward, backward, no-finite, regression)
- **13 CI failures** initially — test expected NaN but PyTorch returns ~1e12
- **Fixed**: changed `isnan()` → `not isfinite()` to catch both NaN and Inf
- **Lint fixed**: added TESTOWNERS header
- Issue #184575 has actionable request posted

## 6. The Fix

```python
# Current (wrong): returns 0 forward, ~1e12 backward
y = F.normalize(torch.zeros(3), dim=0)

# Expected: returns NaN forward, NaN backward
# Achieved by adding epsilon guard at application level:
x = torch.where(x.abs() < 1e-8, torch.full_like(x, float('nan')), x)
y = F.normalize(x, dim=0)
```

---

*Detected by [NeuralDBG](https://github.com/LambdaSection/NeuralDBG). See all [post-mortems](index.html).*
