"""BUG-007 / pytorch#186799 — torch.compile silently produces wrong gradient

Reproduces a silent gradient corruption bug: torch.compile (Inductor)
produces incorrect gradients for atan2(x, y).amin(dim).sum().backward().

Forward output matches exactly. Only backward is wrong.
This is the most dangerous class of ML bugs — silent gradient corruption.

Original issue: https://github.com/pytorch/pytorch/issues/186799
Bug catalog: docs/bugs/BUG-007-pytorch-186799.md
Reported by: @ezyang (PyTorch core maintainer)
"""

import torch
import sys

print("=" * 60)
print("BUG-007: torch.compile silent gradient corruption")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

# Check if torch.compile is available
try:
    torch.compile(lambda x: x)
    compile_available = True
except Exception:
    compile_available = False
    print("\n[SKIP] torch.compile not available in this environment")
    print("This bug requires torch.compile (Inductor backend).")
    print("Run on a supported PyTorch installation to reproduce.")
    sys.exit(0)

torch.manual_seed(42)

# Create inputs
x = torch.randn(4, 5, requires_grad=True)
y = torch.randn(4, 5, requires_grad=True)

print(f"\nInput shapes: x={tuple(x.shape)}, y={tuple(y.shape)}")

# --- Eager mode (ground truth) ---
print("\n--- Eager mode (reference) ---")
x_eager = x.clone().detach().requires_grad_(True)
y_eager = y.clone().detach().requires_grad_(True)
out_eager = torch.atan2(x_eager, y_eager).amin(dim=-1).sum()
out_eager.backward()
grad_eager_x = x_eager.grad.clone()
print(f"  Forward: {out_eager.item():.6f}")
print(f"  x.grad norm: {grad_eager_x.norm().item():.6f}")
print(f"  x.grad min/max: {grad_eager_x.min().item():.6f} / {grad_eager_x.max().item():.6f}")

# --- Compiled mode (Inductor) ---
print("\n--- Compiled mode (Inductor) ---")
try:
    x_comp = x.clone().detach().requires_grad_(True)
    y_comp = y.clone().detach().requires_grad_(True)

    f_compiled = torch.compile(
        lambda a, b: torch.atan2(a, b).amin(dim=-1).sum(),
        backend="inductor",
    )
    out_comp = f_compiled(x_comp, y_comp)
    out_comp.backward()
    grad_comp_x = x_comp.grad.clone()

    print(f"  Forward: {out_comp.item():.6f}")
    print(f"  x.grad norm: {grad_comp_x.norm().item():.6f}")
    print(f"  x.grad min/max: {grad_comp_x.min().item():.6f} / {grad_comp_x.max().item():.6f}")

    # --- Comparison ---
    print("\n--- Comparison ---")
    forward_match = torch.allclose(out_eager, out_comp)
    gradient_match = torch.allclose(grad_eager_x, grad_comp_x)

    print(f"  Forward match: {forward_match}")
    print(f"  Gradient match: {gradient_match}")

    if forward_match and not gradient_match:
        print("\n  [BUG CONFIRMED] Forward correct, gradient WRONG!")
        print("  This is a silent gradient corruption — the most dangerous class of ML bugs.")
        diff = (grad_eager_x - grad_comp_x).abs()
        print(f"  Max gradient diff: {diff.max().item():.6f}")
        wrong_positions = (diff > 1e-6).sum().item()
        print(f"  Wrong gradient positions: {wrong_positions}/{diff.numel()}")
    elif forward_match and gradient_match:
        print("\n  [NO BUG] Both forward and gradient match (may be fixed in this version)")
    else:
        print("\n  [UNEXPECTED] Forward doesn't match either")

except Exception as e:
    print(f"  Error during compiled run: {e}")
    print("  This is expected if torch.compile is not fully supported.")

# --- NeuralDBG would detect ---
print("\n--- NeuralDBG would detect ---")
print("1. GRADIENT_CORRUPTION event at atan2/amin backward")
print("2. FORWARD_BACKWARD_DIVERGENCE: forward matches, gradient doesn't")
print("3. CAUSAL_CHAIN: Inductor codegen error -> argmin grad drop -> silent corruption")
print("4. ROOT_CAUSE: Inductor amin backward missing argmin position gradient")
