# BUG-009 — PyTorch #187227 SDPA 32-bit offset overflow in mem-efficient attention

> **MID**: BUG-009
> **Status**: Cataloged
> **Date opened**: 2026-06-24
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/187227
- Title: *"[SDPA] Fix 32-bit offset overflow in mem-efficient attention forward with attn_bias"*
- Status: OPEN, labeled `module: cuda`, `module: correctness (silent)`, `module: sdpa`
- Created: 2026-06-13 (11 days ago)
- Comments: 2

## Root cause

SDPA (Scaled Dot-Product Attention) with `attn_bias` triggers a 32-bit integer overflow in the offset calculation of the mem-efficient attention kernel. This causes silent incorrect results for large tensors (total elements > 2^31).

## Why this matters for NeuralDBG

1. **Silent correctness bug**: No crash, no NaN — just wrong attention outputs
2. **Affects all transformer models** using SDPA with attention bias
3. **Hard to detect**: The bug only manifests for large tensors (near int32 overflow)
4. **NeuralDBG can detect**: Compare eager vs SDPA attention outputs; flag discrepancies

## Deliverables

- [x] BUG-009 tracking file
- [ ] Repro script
- [ ] Upstream comment
