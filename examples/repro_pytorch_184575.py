"""BUG-008 / pytorch#184575 — F.normalize silent gradient corruption at zero input

F.normalize(x) computes x / ||x||. When x = 0, the forward correctly returns
NaN, but the backward returns a finite gradient of ~1e12 instead of NaN.
This is a silent gradient corruption — users see NaN output but finite gradients.

Original issue: https://github.com/pytorch/pytorch/issues/184575
Bug catalog: docs/bugs/BUG-008-pytorch-184575.md

Reproducible on CPU — no GPU needed.
"""

import torch
import torch.nn.functional as F

print("=" * 60)
print("BUG-008: F.normalize gradient corruption at zero input")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

# Test 1: Zero vector
print("\n--- Test 1: Zero vector ---")
x = torch.zeros(3, requires_grad=True)
y = F.normalize(x, dim=0)
y.sum().backward()

print(f"  Input:    {x.data}")
print(f"  Forward:  {y.data}")
print(f"  Forward NaN: {torch.isnan(y).any().item()}")
print(f"  Gradient: {x.grad}")
print(f"  Gradient NaN: {torch.isnan(x.grad).any().item()}")
print(f"  Gradient max: {x.grad.abs().max().item():.3e}")

forward_has_nan = torch.isnan(y).any().item()
grad_has_nan = torch.isnan(x.grad).any().item()
grad_is_finite = x.grad.isfinite().all().item()

if forward_has_nan and not grad_has_nan:
    print("\n  [BUG CONFIRMED] Forward NaN but backward returns finite gradient!")
    print(f"  Expected: NaN gradient when forward is NaN")
    print(f"  Got: finite gradient of ~{x.grad.abs().max().item():.1e}")
elif forward_has_nan and grad_has_nan:
    print("\n  [FIXED] Both forward and backward return NaN")
else:
    print(f"\n  [UNEXPECTED] Forward NaN: {forward_has_nan}, Grad NaN: {grad_has_nan}")

# Test 2: Near-zero vector (edge case)
print("\n--- Test 2: Near-zero vector ---")
x2 = torch.tensor([1e-8, 1e-8, 1e-8], requires_grad=True)
y2 = F.normalize(x2, dim=0)
y2.sum().backward()
print(f"  Forward NaN: {torch.isnan(y2).any().item()}")
print(f"  Gradient max: {x2.grad.abs().max().item():.3e}")
print(f"  Gradient finite: {x2.grad.isfinite().all().item()}")

# Test 3: Normal input (should work fine)
print("\n--- Test 3: Normal input (control) ---")
x3 = torch.randn(3, requires_grad=True)
y3 = F.normalize(x3, dim=0)
y3.sum().backward()
print(f"  Forward NaN: {torch.isnan(y3).any().item()}")
print(f"  Gradient NaN: {torch.isnan(x3.grad).any().item()}")
print(f"  Gradient finite: {x3.grad.isfinite().all().item()}")

# NeuralDBG would detect
print("\n--- NeuralDBG would detect ---")
print("1. GRADIENT_NORM_SPIKE: ~1e12 at normalize layer")
print("2. FORWARD_BACKWARD_MISMATCH: forward NaN, backward finite")
print("3. CAUSAL_CHAIN: zero input -> norm=0 -> NaN forward -> wrong backward -> 1e12 grad")
print("4. ROOT_CAUSE: normalize backward missing zero-norm guard")
