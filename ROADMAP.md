# NeuralDBG Roadmap — MVP Phase

**Duration**: 5 weeks (Feb 25 - Mar 31, 2026)
**Progress Start**: 10% (Mom Test Complete)
**Progress Current**: 62% (Pessimistic - Updated 2026-04-16)

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

## Progress Calculation

| Component | Weight | Status |
|-----------|--------|--------|
| Mom Test | 10% | Complete (10%) |
| Core functionality | 40% | 4 event types implemented, collapse done (32%) |
| Test coverage (60%+) | 20% | Complete -- 72 tests, 85%+ coverage (20%) |
| Security hardening | 10% | bandit 0 issues (8%) |
| CI/CD & DevOps | 10% | Pipelines configured (8%) |
| Documentation | 10% | README, INFERENCE_FLOW, CODEBASE_GUIDE done (8%) |

**Current Progress**: 10 + 32 + 20 + 8 + 8 + 8 = **86%** raw, pessimistic multiplier 0.72 = **62%**

Pessimistic deductions:
- Dogfooding not done yet on a real model (Rule 58)
- No version tag created yet (Rule 19)
- Profile README not synced (Rule 51)
- Robustness/scale testing still TODO

---

## Next Phase (Post-MVP)
*Only plan after MVP truth is established*
- Research feedback integration
- Formalize inference semantics
- Expand causal question types
- Explanation visualization (not tensor visualization)
