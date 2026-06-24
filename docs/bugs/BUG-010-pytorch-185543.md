# BUG-010 — PyTorch #185543 Inductor gradient mismatch for torch.quantile on tied values

> **MID**: BUG-010
> **Status**: Cataloged
> **Date opened**: 2026-06-24
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/185543
- Title: *"[inductor] Gradient mismatch between eager and inductor for torch.quantile on tied values"*
- Status: OPEN, labeled `triaged`, `module: correctness (silent)`, `oncall: pt2`
- Created: 2026-05-28 (27 days ago)
- Comments: 1

## Root cause

`torch.quantile` under `torch.compile` (Inductor backend) produces incorrect gradients when input contains tied (duplicate) values. The eager mode gradient is correct; the compiled mode gradient differs silently.

## Why this matters for NeuralDBG

1. **Gradient mismatch**: Eager vs compiled divergence — core NeuralDBG detection domain
2. **Silent**: No crash, no NaN — just wrong gradients
3. **Statistical functions**: quantile is used in normalization, outlier detection, robust statistics
4. **27 days old, 1 comment, no PR**: Open for contribution

## Deliverables

- [x] BUG-010 tracking file
- [ ] Repro script
- [ ] Upstream comment
