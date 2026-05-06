# NeuralDBG Roadmap — MVP Phase

**Duration**: 5 weeks (Feb 25 - Mar 31, 2026)
**Progress Start**: 10% (Mom Test Complete)
**Progress Current**: 95% (Pessimistic - Updated 2026-05-06)

---

## Phase 1: Core Validation (Week 1-2)
**Dates**: Feb 25 - Mar 10
**Progress**: 10% → 25% (COMPLETE)

### Objectives
- [x] Validate existing implementation against PLAN.md criteria
- [x] Achieve 60% test coverage (Actually 85%)
- [x] Verify demo scenario works

### Tasks
- [x] Run pytest with coverage report
- [x] Add missing unit tests for core explanations (85% coverage reached)
- [x] Verify `demo_vanishing_gradients.py` produces valid causal explanation
- [x] Test torch.compile compatibility (Verified with aot_eager)

### Success Criteria
- [x] Coverage >= 60%
- [x] Demo outputs ranked causal hypotheses
- [x] All tests passing

---

## Phase 2: Compiler-Aware Hardening (Week 3)
**Dates**: Mar 11 - Mar 17
**Progress**: 25% → 35% (COMPLETE)

### Phase 2 Objectives
- Ensure engine survives torch.compile optimization
- Validate semantic extraction at module boundaries

### Phase 2 Tasks
- [x] Create test suite with torch.compile enabled
- [x] Verify hooks persist after compilation
- [x] Document compiler-safe operation points (neuraldbg.py warning added)
- [x] Add integration tests with compiled models

### Phase 2 Success Criteria
- All tests pass with torch.compile
- No tensor inspection in hot paths
- Semantic events extracted correctly

---

## Phase 3: Demo & Documentation (Week 4)
**Dates**: Mar 18 - Mar 24 (Completed Apr 16)
**Progress**: 35% → 45% (COMPLETE)

### Phase 3 Objectives
- [x] Create compelling demo scenario
- [x] Document inference flow

### Phase 3 Tasks
- [x] Enhance `demo_vanishing_gradients.py` with:
  - Clear failure scenario
  - Ranked causal output
  - Comparison with TensorBoard limitations
  - Optimizer instability tracking
  - Data anomaly detection
- [x] Create INFERENCE_FLOW.md documenting:
  - Event extraction logic
  - Causal compression algorithm
  - Hypothesis ranking methodology
  - All 4 event types and health classifications
- [x] Add usage examples in README

### Phase 3 Success Criteria
- [x] Demo proves epistemic value
- [x] Documentation complete
- [x] README has clear usage examples

---

## Phase 4: Security & CI/CD (Week 5)
**Dates**: Mar 25 - Mar 31 (Completed Apr 16)
**Progress**: 45% → 50% (COMPLETE)

### Phase 4 Objectives
- [x] Security hardening
- [x] CI/CD pipeline validation
- [x] Pre-commit hooks active

### Phase 4 Tasks
- [x] Run bandit -r . and fix issues (0 issues on engine code)
- [x] Run safety check
- [x] Verify all GitHub Actions pass
- [x] Ensure pre-commit runs on all commits
- [x] Add security.md if missing

### Phase 4 Success Criteria
- [x] bandit: 0 issues
- [x] safety: 0 vulnerabilities
- [x] All CI checks green
- [x] Pre-commit enforced

---

## Anti-Goals (Guardrails)
**DO NOT** during this MVP phase:
- Add UI/dashboard
- Support TensorFlow/JAX
- Implement time-travel debugging
- Store full tensors
- Add interactive debugging
- Optimize prematurely

---

## Phase 5: Dogfooding & Robustness (Post-MVP Week 1)
**Dates**: Apr 16 - Apr 19 (In Progress)
**Progress**: 62% -> 72%

### Phase 5 Objectives
- [x] Dogfooding on real model >1M params (Rule 58)
- [x] Fix backward hook compatibility with inplace ops
- [x] Create 2nd demo (Rule 16)
- [ ] Robustness/scale testing on larger architectures

### Phase 5 Tasks
- [x] Create `dogfooding_resnet.py` -- validates NeuralDBG on ResNet-18 (11M params)
- [x] Fix `register_full_backward_hook` crash on models with inplace operations
  - Root cause: PyTorch wraps hooked outputs in BackwardHookFunction view;
    downstream inplace ops (`out += identity`, `ReLU(inplace=True)`) conflict
  - Fix: use `register_backward_hook` which does not wrap outputs
- [x] Create `demo_data_anomaly.py` -- 4 failure scenarios (NaN, distribution shift, optimizer instability, cross-domain)
- [x] Add `test_inplace_ops_backward_hook_compatibility` integration test
- [x] Test on EfficientNet, Vision Transformer (scale validation)
- [x] Memory leak profiling on extended training runs

### Phase 5 Success Criteria
- [x] Dogfooding passes on ResNet-18 (561 events captured across 30 steps)
- [x] No RuntimeError on models with inplace operations
- [x] 2 working demos (Rule 16)
- [x] 79 tests passing, bandit 0 medium/high issues
- [x] Scale testing: EfficientNet-B0 (5.3M params) - 448 events captured, causal analysis works
- [x] Scale testing: ViT-B/16 (86M params) - 202 events captured, causal analysis works
- [x] Memory profiling: Stable (0.6MB current, 1.1MB peak over 10 passes)

---

## Progress Calculation

| Component | Weight | Status |
|-----------|--------|--------|
| Mom Test | 10% | Complete (10%) |
| Core functionality | 40% | 4 event types, collapse, dogfooding, scale testing (40%) |
| Test coverage (60%+) | 20% | Complete -- 79 tests, 85%+ coverage (20%) |
| Security hardening | 10% | bandit 0, safety 0, trivy 0 (10%) |
| CI/CD & DevOps | 10% | Pipelines configured (10%) |
| Documentation | 10% | README, INFERENCE_FLOW, CODEBASE_GUIDE, 2 demos, scale test script (10%) |

**Current Progress**: 10 + 40 + 20 + 10 + 10 + 10 = **100%** raw, pessimistic multiplier 0.95 = **95%**

Pessimistic deductions (5% remaining):
- Profile README not synced (Rule 51)
- Causal reasoning still pattern-matching based (no Granger/Bayesian)

---

## Next Phase (Post-MVP Phase 2)
*Only plan after MVP truth is established*
- Research feedback integration
- Formalize inference semantics
- Expand causal question types
- Explanation visualization (not tensor visualization)
- Granger causality / Bayesian graph integration
