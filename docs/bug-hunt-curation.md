# M2 Bug-Hunt Curation — Next 6 candidates

> Curated 2026-06-09. Status as of today: **4/10** bugs done (BUG-001..004).
> Need **6 more** for M2 (10 real bugs). This document rates candidates against
> the [bug-hunt charter](file:///c:/Users/Utilisateur/Documents/NeuralDBG/docs/bug_hunt_charter.md).
>
> **Already chased this session**: BUG-005 = [pytorch#173334](https://github.com/pytorch/pytorch/issues/173334) (LSTM batch pollution, BUG-005 created, comment drafted).

## Score legend
- **Fit**: how well the bug fits NeuralDBG's core (NaN/vanishing/exploding/collapse) + repro on consumer GPU
- **Reach**: comment count / stars / activity (community validation)
- **Ease**: how hard to reproduce + write a NeuralDBG detector
- **No-PR**: is there an open PR for this? (no = good, gives us a chance to contribute)
- **Total**: weighted score, 1-5, **bigger = better**

## Candidate shortlist (in priority order)

| # | Issue | Title | Fit | Reach | Ease | No-PR | Total | Why |
|---|-------|-------|:---:|:-----:|:----:|:-----:|:-----:|-----|
| 6 | [pytorch#185912](https://github.com/pytorch/pytorch/issues/185912) | Non-finite gradient with extremely low learning rate (FSDP+AMP) | 5 | 4 | 4 | ✅ | **4.3** | Modern stack (FSDP+AMP). Counter-intuitive failure (low LR shouldn't break). Easy to wrap with `autocast` injection. |
| 7 | [pytorch#174011](https://github.com/pytorch/pytorch/issues/174011) | LayerNorm NaN on CPU but normal on CUDA | 5 | 3 | 5 | ✅ | **4.3** | Fundamental numerical stability. Same author as LSTM bugs. Easy synthetic repro. LayerNorm is in every transformer. |
| 8 | [huggingface#43844](https://github.com/huggingface/transformers/issues/43844) | HfDeepSpeedConfig + ZeRO-3 + random init → gradient explosion | 5 | 3 | 4 | ✅ | **4.0** | Modern training stack (DeepSpeed ZeRO-3). Bug in `Trainer` initialization. Reproducible on consumer hardware with 1.1B model. |
| 9 | [pytorch#181555](https://github.com/pytorch/pytorch/issues/181555) | CUDA layer_norm wrong output when flattened size > 2^32 | 4 | 3 | 3 | ✅ (PR #186582 in progress, but not merged) | **3.3** | Specific edge case (large tensors). Good defensive content for `detect_large_tensor_anomaly`. |
| 10 | [pytorch#178084](https://github.com/pytorch/pytorch/issues/178084) | `torch.compile` introduces NaN in LayerNorm on valid float32 boundary | 4 | 2 | 4 | ✅ (PR #186582 may fix it indirectly) | **3.3** | Compile vs eager divergence. Important for any user of `torch.compile`. |
| 11 | [pytorch#173927](https://github.com/pytorch/pytorch/issues/173927) | LSTM CUDA vs CPU divergence 200% (1.52 abs diff) | 4 | 2 | 3 | ✅ | **3.0** | Sample-independence cousin. Lower-priority than BUG-005. |

## Not recommended (lower score)

- **PR#8035 (DeepSpeed)** — already has a PR with a fix proposed; we can't beat it
- **PR#186582 (persistent Welford)** — already in flight by `jansel`, core maintainer territory
- Issues closed since 2026-04 — M2 needs OPEN bugs per charter

## How to chase a bug (workflow)

1. Pick a number (#6 through #11 above)
2. Create `docs/bugs/BUG-00N-<system>-<id>.md` following [BUG-005](file:///c:/Users/Utilisateur/Documents/NeuralDBG/docs/bugs/BUG-005-pytorch-173334.md) template
3. Write `examples/repro_<system>_<id>.py` — self-contained, no large downloads
4. (Optional but high-value) Write `tests/unit/test_<short-name>_detection.py` — CI-friendly injection-based test
5. Write `docs/posts/<system>_<id>_comment.md` — draft upstream comment (CEO TODO: manual post)
6. Commit per pattern: `feat(bug-00N): add <system>/<id> catalog + repro + comment`
7. Add a row to the upstream PR tracker in [ROADMAP.md](file:///c:/Users/Utilisateur/Documents/NeuralDBG/ROADMAP.md)

## Effort estimate per bug

| Phase | Time |
|-------|------|
| Catalog entry (markdown) | 30 min |
| Self-contained repro script | 1–2h |
| CI-friendly detection test | 1h (if bug is hardware-agnostic) or skipped (if hardware-only) |
| Upstream comment draft | 15 min |
| Manual post + tracking | 5 min |
| **Total per bug** | **~2–4h** |
| **For 6 more bugs** | **~12–24h, 1.5–3 days** |

Realistic timeline: ~1 bug per day for 6 days → M2 done by mid-June. Track
in [PLAN.md](file:///c:/Users/Utilisateur/Documents/NeuralDBG/PLAN.md) (private).
