# BUG-008 — PyTorch #184575 F.normalize silent gradient corruption at zero input

> **MID**: BUG-008
> **Status**: Cataloged
> **Date opened**: 2026-06-24
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/184575
- Title: *"F.normalize returns ~1e12 gradient at zero-vector input instead of NaN"*
- Status upstream: OPEN, labeled `module: autograd`, `triaged`, `module: norms and normalization`
- Created: 2026-05-20 (35 days ago)
- Comments: 1

## Root cause

`F.normalize(x, dim)` computes `x / ||x||`. When `x = 0`, the norm is 0, and the result is mathematically undefined. The **forward pass correctly returns NaN**, but the **backward pass returns a finite gradient of ~1e12** instead of NaN.

This is a silent gradient corruption — the user sees NaN in their output but finite gradients, making debugging extremely confusing. The PyTorch autograd documentation states that undefined-input gradients should be NaN.

## Why this is PERFECT for NeuralDBG

1. **Forward/backward inconsistency**: Forward returns NaN, backward returns 1e12 — NeuralDBG's gradient health events catch this mismatch
2. **Simple operation**: `F.normalize` — used everywhere (attention, embeddings, etc.)
3. **Silent corruption**: Not a crash, not a NaN — just wrong values that corrupt training
4. **Reproducible on CPU**: No GPU needed
5. **35 days old, 1 comment, no PR**: Wide open for contribution

## Minimal reproduction

```python
import torch
import torch.nn.functional as F

x = torch.zeros(3, requires_grad=True)
y = F.normalize(x, dim=0)
y.sum().backward()

print(f"Forward NaN: {torch.isnan(y).any().item()}")     # True
print(f"Gradient max: {x.grad.abs().max().item():.3e}")  # ~1e12
print(f"Gradient NaN: {torch.isnan(x.grad).any().item()}") # False (BUG!)
```

## NeuralDBG detection

1. **gradient_norm_spike** event: gradient norm ~1e12 at normalize layer
2. **forward_backward_mismatch**: forward produces NaN, backward produces finite
3. **causal_chain**: zero input → norm = 0 → division by zero → NaN forward → wrong backward formula → 1e12 gradient

## Proposed fix

Fix the backward formula for `F.normalize` to propagate NaN when the input is zero (norm = 0). The backward should check for zero-norm inputs and return NaN instead of computing the finite-difference approximation.

## Deliverables

- [x] BUG-008 tracking file
- [ ] `examples/repro_pytorch_184575.py`
- [ ] Upstream comment with NeuralDBG diagnostic
- [ ] PR with fix
