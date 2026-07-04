# POST-002 — The Silent NaN Factory: varlen_attn and the Padding Problem

> **BUG-002** | **Source**: pytorch/pytorch#176793 | **PR**: #188933 (REAL FIX, pas juste un test)
> **Date**: 2026-07-04 | **Detection**: NeuralDBG | **Fix type**: Input validation (ValueError)

---

## 1. The Bug

**What**: `varlen_attn()` silently produces NaN gradients when query/key tensors are longer than `cu_seqlens[-1]`. Forward pass completes without error. Backward produces NaN.

**Upstream**: [pytorch#176793](https://github.com/pytorch/pytorch/issues/176793) — OPEN.

This is a **silent correctness bug**: extra padding tokens (common in batched variable-length sequences) cause NaN gradients with no error message. Users debug for hours because the forward pass looks completely fine.

## 2. Reproduction

```python
import torch

TOTAL = 944
cu_seqlens = torch.tensor([0, 144, 432, 944], dtype=torch.int32, device='cuda')
x = torch.randn(TOTAL + 2, 1024, device='cuda', requires_grad=True)
# 2 extra padding tokens — forward OK, backward NaN!
q, k, v = x.chunk(3, dim=-1)

loss = torch.nn.attention.varlen.varlen_attn(
    q, k, v, cu_seqlens, cu_seqlens, 512, 512
).sum()
loss.backward()  # NaN in gradients!
```

## 3. Why It Happens

The extra tokens are outside the attention computation (not covered by any sequence in `cu_seqlens`) but still participate in the autograd graph. The forward pass ignores them, but the backward pass tries to compute gradients through them — and fails.

## 4. NeuralDBG Diagnosis

NeuralDBG traces the NaN gradient back to `varlen_attn`:
```
gradient_health_transition at qkv.weight: NORMAL → nan_detected
  → optimizer_instability at Adam: diverging
  → training failure
```

The causal chain pinpoints `qkv.weight` as the source of NaN.

## 5. The Fix (PR #188933)

Instead of silently producing NaN, we raise a clear `ValueError`:

```python
total_q = cu_seq_q[-1].item()
if query.size(0) > total_q:
    raise ValueError(
        f"query has {query.size(0)} tokens but cu_seq_q[-1] = {total_q}. "
        f"query length must not exceed cu_seq_q[-1] to avoid NaN gradients. "
        f"See https://github.com/pytorch/pytorch/issues/176793."
    )
```

**This converts a silent NaN corruption into an explicit, actionable error.** The user immediately knows what's wrong and how to fix it.

## 6. Why This Is Different

Unlike PRs #188053, #188066, #188923 (test-only), this PR changes **core PyTorch behavior** in `torch/nn/attention/varlen.py`. It's a real fix, not just detection.

---

*Detected by [NeuralDBG](https://github.com/LambdaSection/NeuralDBG). See all [post-mortems](index.html).*
