"""BUG-006 / pytorch#187759 — torch.linalg.svdvals silently swallows NaN

Reproduces a silent correctness bug: svdvals() returns finite values when
given a NaN matrix, while svd() correctly propagates NaN.

NOTE: Behavior varies by PyTorch version and LAPACK backend:
  - PyTorch 2.11.0 CPU: Both raise RuntimeError (partial fix)
  - Earlier versions / CUDA: svdvals returns finite values (the bug)
  - See issue for backend-specific behavior

Original issue: https://github.com/pytorch/pytorch/issues/187759
Bug catalog: docs/bugs/BUG-006-pytorch-187759.md
"""

import sys
import torch

print("=" * 60)
print("BUG-006: torch.linalg.svdvals NaN swallowing")
print(f"PyTorch: {torch.__version__} | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("=" * 60)

# Create a matrix with NaN
A = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, float('nan'), 6.0],
                   [7.0, 8.0, 9.0]])

print(f"\nInput matrix A:\n{A}")
print(f"Contains NaN: {torch.isnan(A).any().item()}")

# Test svdvals
print("\n--- svdvals ---")
try:
    result = torch.linalg.svdvals(A)
    has_nan = torch.isnan(result).any().item()
    print(f"svdvals(A) = {result}")
    print(f"  Contains NaN: {has_nan}")
    if not has_nan:
        print("  [BUG] svdvals returned FINITE values for NaN input!")
    else:
        print("  [OK] svdvals correctly returned NaN")
except RuntimeError as e:
    print(f"  RuntimeError (fixed in this version): {str(e)[:80]}")

# Test svd for comparison
print("\n--- svd (reference) ---")
try:
    U, S, Vh = torch.linalg.svd(A)
    print(f"svd(A).S = {S}")
    print(f"  Contains NaN: {torch.isnan(S).any().item()}")
except RuntimeError as e:
    print(f"  RuntimeError: {str(e)[:80]}")

# If no bug on this platform, try CUDA if available
if torch.cuda.is_available():
    print("\n--- Testing on CUDA ---")
    A_cuda = A.cuda()
    try:
        result_cuda = torch.linalg.svdvals(A_cuda)
        has_nan_cuda = torch.isnan(result_cuda).any().item()
        print(f"svdvals(A) on CUDA = {result_cuda}")
        print(f"  Contains NaN: {has_nan_cuda}")
        if not has_nan_cuda:
            print("  [BUG CONFIRMED ON CUDA] svdvals returned finite values!")
    except RuntimeError as e:
        print(f"  RuntimeError on CUDA: {str(e)[:80]}")

print("\n--- NeuralDBG would detect ---")
print("1. DATA_ANOMALY: NaN in input tensor")
print("2. SVD_OUTPUT_VALID: svdvals returned finite despite NaN input")
print("3. CAUSAL_CHAIN: NaN -> svdvals swallow -> matrix_rank false positive")
print("4. ROOT_CAUSE: svdvals missing/inconsistent NaN guard vs backend")
