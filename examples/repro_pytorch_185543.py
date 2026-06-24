"""BUG-010 / pytorch#185543 — Inductor gradient mismatch for torch.quantile on tied values

torch.quantile under torch.compile produces incorrect gradients when input
contains tied (duplicate) values. Eager mode is correct; compiled mode differs silently.

NOTE: Requires torch.compile (Inductor backend). Our setup doesn't support it.
This script documents the pattern.

Original issue: https://github.com/pytorch/pytorch/issues/185543
Bug catalog: docs/bugs/BUG-010-pytorch-185543.md
"""

import torch
import sys

print("=" * 60)
print("BUG-010: quantile gradient mismatch under torch.compile")
print(f"PyTorch: {torch.__version__}")
print("=" * 60)

# Check compile availability
try:
    torch.compile(lambda x: x)
    compile_ok = True
except Exception:
    compile_ok = False

if not compile_ok:
    print("\n[SKIP] torch.compile not available.")
    print("Documenting expected failure mode.")

print("""
Expected failure mode:
1. Create tensor with tied (duplicate) values
2. Compute torch.quantile in eager mode → correct gradient
3. Compute torch.quantile under torch.compile → wrong gradient
4. Silent gradient mismatch — no crash, no NaN

Minimal repro (when compile is available):
    x = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0], requires_grad=True)
    # Eager (correct)
    q_eager = torch.quantile(x, 0.5)
    q_eager.backward()
    grad_eager = x.grad.clone()
    x.grad = None
    # Compiled (BUG: wrong gradient)
    f = torch.compile(lambda t: torch.quantile(t, 0.5))
    q_comp = f(x)
    q_comp.backward()
    grad_comp = x.grad.clone()
    # Compare — should be equal, but aren't

NeuralDBG would detect:
1. GRADIENT_MISMATCH: eager vs compiled gradients differ
2. COMPILE_DIVERGENCE event at quantile operation
3. CAUSAL_CHAIN: tied values -> inductor grad bug -> silent training corruption
""")
