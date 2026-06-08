---
name: Upstream PR with NeuralDBG Solution
about: Template for PRs submitted to upstream repos (PyTorch, HuggingFace, etc.) with NeuralDBG diagnostic evidence
title: '[BUG-XXX] Short description'
labels: 'upstream, needs-review'
assignees: ''
---

## Summary

<!-- What bug does this PR fix? Link to upstream issue. -->

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

---
**Note:** This PR includes NeuralDBG diagnostic evidence to help reviewers understand the root cause. The diagnostic output is from a real run, not synthesized.
