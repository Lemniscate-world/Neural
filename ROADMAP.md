# NeuralDBG Roadmap — MVP Phase

**Duration**: 5 weeks (Feb 25 - Mar 31, 2026)
**Progress Start**: 10% (Mom Test Complete)
**Progress Current**: 51% (Extreme Pessimistic - Corrected 2026-03-21)

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

### Objectives
- Ensure engine survives torch.compile optimization
- Validate semantic extraction at module boundaries

### Tasks
- [x] Create test suite with torch.compile enabled
- [x] Verify hooks persist after compilation
- [x] Document compiler-safe operation points (neuraldbg.py warning added)
- [x] Add integration tests with compiled models

### Success Criteria
- All tests pass with torch.compile
- No tensor inspection in hot paths
- Semantic events extracted correctly

---

## Phase 3: Demo & Documentation (Week 4)
**Dates**: Mar 18 - Mar 24
**Progress**: 35% → 45%

### Objectives
- Create compelling demo scenario
- Document inference flow

### Tasks
- [ ] Enhance `demo_vanishing_gradients.py` with:
  - Clear failure scenario
  - Ranked causal output
  - Comparison with TensorBoard limitations
- [ ] Create INFERENCE_FLOW.md documenting:
  - Event extraction logic
  - Causal compression algorithm
  - Hypothesis ranking methodology
- [ ] Add usage examples in README

### Success Criteria
- Demo proves epistemic value
- Documentation complete
- README has clear usage examples

---

## Phase 4: Security & CI/CD (Week 5)
**Dates**: Mar 25 - Mar 31
**Progress**: 45% → 50%

### Objectives
- Security hardening
- CI/CD pipeline validation
- Pre-commit hooks active

### Tasks
- [ ] Run bandit -r . and fix issues
- [ ] Run safety check
- [ ] Verify all GitHub Actions pass
- [ ] Ensure pre-commit runs on all commits
- [ ] Add security.md if missing

### Success Criteria
- bandit: 0 issues
- safety: 0 vulnerabilities
- All CI checks green
- Pre-commit enforced

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
| Mom Test | 10% | ✅ Complete |
| Core functionality | 40% | 🔄 In Progress (30%) |
| Test coverage (60%+) | 20% | ✅ Complete (85% reached) |
| Security hardening | 10% | ⚠️ Partial (5%) |
| CI/CD & DevOps | 10% | ✅ Done |
| Documentation | 10% | ⚠️ Partial (5%) |

**Current Progress**: 10% (Mom) + 10%(Ext) + 10%(Sem) + 10%(Reas) + 0%(Rob) + 20%(Test) + 5%(Sec) + 8%(Dev) + 3%(Doc) = **76%** (pessimistic: **51%**)

---

## Next Phase (Post-MVP)
*Only plan after MVP truth is established*
- Research feedback integration
- Formalize inference semantics
- Expand causal question types
- Explanation visualization (not tensor visualization)
