---
name: Upstream PR with NeuralDBG Solution
about: Template for PRs submitted to upstream repos (PyTorch, HuggingFace, etc.) with NeuralDBG diagnostic evidence
title: '[BUG-XXX] Short description'
labels: 'upstream, needs-review'
assignees: ''
---

> ⚠️ **MANDATORY**: You MUST pass ALL 6 gates in [PR_GATE.md](../PR_GATE.md) before using this template.
> If you haven't read PR_GATE.md, STOP and read it now.

## Summary

<!-- What bug does this PR fix? Link to upstream issue. -->

## PR Gate Confirmation

<!-- Check each box after verifying in PR_GATE.md -->
- [ ] **G1 FIX**: This is a behavior fix, NOT a warning/log/print
- [ ] **G2 EXISTS**: PR is open on upstream repo (this one!)
- [ ] **G3 REPRO**: Reproduction runs on CPU or free GPU
- [ ] **G4 CITE**: NeuralDBG cited below with detection evidence
- [ ] **G5 FOLLOW-UP**: Plan defined (see bottom of this PR)
- [ ] **G6 NO-WORKAROUND**: grep -ri "workaround" returns 0 matches

## NeuralDBG Diagnostic Evidence

<!-- How did NeuralDBG detect and localize this bug? Include causal chain. -->

### Detection

```
# Paste NeuralDBG output here
dbg.explain_failure()
```

### Causal Chain

| Step | Event | Layer | Value |
|------|-------|-------|-------|
| | | | |

### Reproduction

```bash
pip install neuraldbg
python examples/repro_XXXX.py
```

## Fix Description

<!-- What does this PR change? Why does it fix the bug? -->

## Verification

- [ ] Bug reproduces without fix
- [ ] Bug is fixed with this PR
- [ ] NeuralDBG detects the fix (no more NaN/gradient events)
- [ ] Existing tests pass
- [ ] New tests added for this specific failure mode

## Upstream Issue Link

<!-- Link to the original issue: https://github.com/... -->

## Checklist

- [ ] This PR addresses a real upstream bug (not a synthetic scenario)
- [ ] Reproduction script is self-contained and runnable
- [ ] NeuralDBG output is from actual run (not mocked)
- [ ] Fix is minimal and focused on the root cause
- [ ] Documentation updated if needed

## Follow-Up Plan

<!-- MANDATORY per PR_GATE.md Gate 5. What will you do if no review? -->

- **Day 3** (if no review): _________________________________
- **Day 7** (if no review): _________________________________
- **Day 14** (if no review): ________________________________

**Shared on**:
- [ ] X / Twitter
- [ ] PyTorch Forums / HuggingFace Discord
- [ ] Reddit (r/pytorch, r/MachineLearning)

---
**Note:** This PR includes NeuralDBG diagnostic evidence to help reviewers understand the root cause. The diagnostic output is from a real run, not synthesized.
