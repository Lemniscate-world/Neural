# BUG-007 — PyTorch #186799 torch.compile silently produces wrong gradient for atan2

> **MID**: BUG-007
> **Status**: Cataloged
> **Date opened**: 2026-06-24
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/186799
- Title: *"torch.compile silently produces wrong gradient for atan2(x, y).amin(dim) on float32"*
- Status upstream: OPEN, labeled `module: autograd`, `triaged`, `oncall: pt2`, `oncall: cpu inductor`
- Author: @ezyang (PyTorch core maintainer!)
- Created: 2026-06-09 (15 days ago)
- Comments: 0

## Root cause

`torch.compile` (Inductor backend) produces an incorrect gradient for the expression `atan2(x, y).amin(dim=-1).sum().backward()`. The gradient at argmin positions is silently dropped (becomes 0.0 or -0.0) instead of receiving the correct upstream gradient value.

Key characteristics:
1. **Forward pass is CORRECT** — loss values match exactly between eager and compiled
2. **Backward pass is WRONG** — gradient values are silently incorrect at specific positions
3. **Only in Inductor backend** — `aot_eager` is correct
4. **Distinct from known stride-0 bugs** — involves plain atan2 + amin pattern

This is a **silent gradient corruption** — the most dangerous class of bugs because:
- No NaN or Inf (harder to detect)
- Loss curve looks normal (forward is correct)
- Training proceeds but converges to wrong parameters
- Impossible to diagnose without gradient-level instrumentation

## Why this is PERFECT for NeuralDBG

1. **Silent gradient corruption**: NeuralDBG's core value proposition
2. **Forward/backward divergence**: NeuralDBG detects this via gradient health events
3. **Reported by a PyTorch maintainer** (@ezyang): high credibility
4. **0 comments, no PR**: we can be first to contribute
5. **Minimal repro**: single tensor operation chain

## Minimal reproduction

```python
import torch

x = torch.randn(4, 5, requires_grad=True)
y = torch.randn(4, 5, requires_grad=True)

# Eager (correct)
out_eager = torch.atan2(x, y).amin(dim=-1).sum()
out_eager.backward()
grad_eager = x.grad.clone()
x.grad = None

# Compiled (BUG: wrong gradient)
f = torch.compile(lambda a, b: torch.atan2(a, b).amin(dim=-1).sum())
out_comp = f(x, y)
out_comp.backward()
grad_comp = x.grad.clone()

# Compare: they SHOULD be equal but aren't
print("Forward match:", torch.allclose(out_eager, out_comp))
print("Gradient match:", torch.allclose(grad_eager, grad_comp))
# Expected: both True
# Actual: forward=True, gradient=False
```

## NeuralDBG detection strategy

NeuralDBG can detect this class of bug by:
1. Computing gradients in eager mode (reference)
2. Computing gradients in compiled mode (actual)
3. Comparing: if forward matches but gradient doesn't → `gradient_corruption` event
4. Localizing: which operation/layer produced the wrong gradient

## Proposed fix (upstream)

Fix the Inductor codegen for `amin` reduction when combined with `atan2` to correctly propagate gradients to argmin positions.

## Deliverables checklist

- [x] BUG-007 tracking file (this file)
- [ ] `examples/repro_pytorch_186799.py` (minimal repro)
- [ ] Upstream comment with NeuralDBG diagnostic
- [ ] PR with fix or test
- [ ] NeuralDBG detection test

## Mom Test R2

- Minimal reproduction (10 lines of code)
- No external data needed
- Reported by PyTorch maintainer (high credibility)
- Silent correctness bug = core NeuralDBG value
