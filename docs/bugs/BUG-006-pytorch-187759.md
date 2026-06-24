# BUG-006 — PyTorch #187759 torch.linalg.svdvals swallows NaN silently

> **MID**: BUG-006
> **Status**: Cataloged, repro created
> **Date opened**: 2026-06-24
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/187759
- Title: *"torch.linalg.svdvals swallows NaN (returns finite values), disagreeing with torch.linalg.svd; matrix_rank then reports full rank for a NaN matrix"*
- Status upstream: OPEN, labeled `module: correctness (silent)`, `module: NaNs and Infs`, `module: linear algebra`
- Author: @OE-GOD
- Created: 2026-06-20 (4 days ago, FRESH)

## Root cause

`torch.linalg.svdvals()` returns finite singular values when given a matrix containing NaN, instead of propagating NaN. This is a **silent correctness bug**: the function completes without error but returns wrong results.

The cascade:
1. User has a matrix with NaN values (e.g., from a corrupted embedding or unstable computation)
2. `svdvals(matrix)` returns finite values (e.g., `[3.0, 2.0, 1.0]`)
3. `matrix_rank(matrix)` uses these values to report "full rank"
4. User proceeds with downstream computation assuming valid inputs
5. Silent data corruption propagates through the pipeline

`torch.linalg.svd()` (the full SVD) correctly propagates NaN. The inconsistency between `svdvals` and `svd` is the bug.

## Trigger conditions

1. Input matrix contains at least one NaN
2. Call `torch.linalg.svdvals()` (not `torch.linalg.svd()`)
3. The function returns finite singular values instead of NaN

```python
import torch
A = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]])
print(torch.linalg.svdvals(A))  # BUG: returns finite values like [4.5, 1.1]
print(torch.linalg.svd(A).S)     # CORRECT: returns NaN
```

## Why this is PERFECT for NeuralDBG

1. **Silent correctness bug**: No error, no crash — just wrong results
2. **Cascade failure**: svdvals → matrix_rank → downstream decisions
3. **NaN propagation**: Core NeuralDBG detection capability
4. **Fresh bug**: Created June 20, no PR yet — we can be first
5. **Easy fix**: Add `isnan` check at the beginning of `svdvals`
6. **High impact**: SVD is used everywhere (PCA, recommender systems, model compression)

## NeuralDBG detection

NeuralDBG can detect this in user code:
1. **data_anomaly** event: NaN detected in input matrix
2. **svd_output_valid** event: svdvals returned finite values despite NaN input
3. **causal_chain**: NaN input → svdvals silent swallow → matrix_rank false positive → training with corrupted features

## Proposed fix (upstream)

```python
# In torch/linalg.py, linalg_svdvals function:
def linalg_svdvals(A, ...):
    # Add NaN check
    if torch.isnan(A).any():
        return torch.full((min(A.shape[-2:]),), float('nan'), 
                         dtype=A.dtype, device=A.device)
    # ... existing implementation
```

## Reproduction script

`examples/repro_pytorch_187759.py` — minimal repro demonstrating the NaN swallowing.

## Deliverables checklist

- [x] BUG-006 tracking file (this file)
- [ ] `examples/repro_pytorch_187759.py` (minimal repro)
- [ ] Upstream comment with NeuralDBG diagnostic
- [ ] PR with fix (add isnan guard to svdvals)
- [ ] NeuralDBG test: `tests/unit/test_svd_nan_detection.py`

## Mom Test R2

- Minimal reproduction (4 lines of code)
- No external data needed
- Fix is a simple guard clause
- High-impact bug (SVD is foundational)
