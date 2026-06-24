# 🚪 PR_GATE.md — Mandatory Gate Before ANY Upstream PR

> **READ THIS BEFORE CREATING ANY UPSTREAM PR.**
> This gate is MANDATORY per DEV_RULES.md rule D4.
> If you skip it, you WILL repeat the mistakes of PR #186631 and BUG-001.

---

## GATE 1: FIX vs WORKAROUND

```
QUESTION: Does this PR FIX the bug or just DETECT/WARN about it?
```

| If your PR... | Gate Result | Action |
|---------------|-------------|--------|
| Adds `warnings.warn()` | ❌ **REJECTED** | Go back. Write a fix, not a warning. |
| Adds `print()` / logging | ❌ **REJECTED** | Same. Detection without correction = 0 value upstream. |
| Catches an exception and warns | ❌ **REJECTED** | Exception is already a signal. What does your PR ADD? |
| Changes behavior to PREVENT the bug | ✅ **PASS** | Proceed to Gate 2. |
| Adds input validation that raises early | ✅ **PASS** | Proceed to Gate 2. |
| Fixes numerical stability (eps, scaling) | ✅ **PASS** | Proceed to Gate 2. |
| Adds a new test that verifies the fix | ✅ **PASS** | Proceed to Gate 2. |

**Lesson from PR #186631**: A `warnings.warn()` in MultiheadAttention was closed by the CEO because PyTorch maintainers won't merge detection-only PRs. They want BEHAVIOR CHANGES.

---

## GATE 2: UPSTREAM PR EXISTS

```
QUESTION: Is there an ACTUAL pull request open on the upstream repo?
```

| Status | Gate Result | Action |
|--------|-------------|--------|
| No PR created, just local code | ❌ **REJECTED** | Pipeline without PR = 0 credibility. Create the PR NOW. |
| PR created but not pushed | ❌ **REJECTED** | Push it. A local branch doesn't count. |
| PR open on upstream repo | ✅ **PASS** | Proceed to Gate 3. |

**Lesson from BUG-001 v2**: The full NeuralDBG+Agent pipeline was ready, but no upstream PR was ever created. Months of work, zero external proof.

---

## GATE 3: REPRODUCTION IS HARDWARE-INDEPENDENT

```
QUESTION: Can a maintainer reproduce this bug on ANY machine?
```

| Status | Gate Result | Action |
|--------|-------------|--------|
| Requires CUDA GPU (A100, H100) | ⚠️ **WARNING** | Provide Colab/Kaggle notebook OR CPU fallback. |
| Requires MPS (Apple Silicon) | ⚠️ **WARNING** | Use gradient injection test (see D3). |
| Requires specific model (70B+) | ⚠️ **WARNING** | Provide smallest model that reproduces (<1B params). |
| Runs on CPU or T4 GPU | ✅ **PASS** | Proceed to Gate 4. |

**Lesson from BUG-003/004**: Both required hardware we don't have. We created CPU injection tests. That's the pattern.

---

## GATE 4: NEURALDBG IS CITED

```
QUESTION: Does the PR description or commit message mention NeuralDBG?
```

| Status | Gate Result | Action |
|--------|-------------|--------|
| No mention of NeuralDBG | ❌ **REJECTED** | Add detection evidence to PR description. |
| NeuralDBG mentioned in description | ✅ **PASS** | Proceed to Gate 5. |

**Why**: Every merged PR is a backlink. Every backlink is a user. This is our ONLY distribution channel right now.

---

## GATE 5: FOLLOW-UP PLAN EXISTS

```
QUESTION: What will you do if the PR gets NO review after 7 days?
```

| Plan | Gate Result | Action |
|------|-------------|--------|
| No plan | ❌ **REJECTED** | Define at least ONE follow-up action. |
| "Wait and hope" | ❌ **REJECTED** | Hope is not a strategy. |
| Comment politely after 7 days | ✅ **PASS** | Proceed. |
| Share on PyTorch Dev Discussions | ✅ **PASS** | Proceed. |
| @mention a relevant maintainer | ✅ **PASS** | Proceed. |
| Post on X/Reddit linking to PR | ✅ **PASS** | Proceed. |

**Lesson from PR #186786**: 16 days, 0 reviews. A good PR without follow-up = invisible.

---

## GATE 6: WORKAROUND WORD BAN (D2 compliance)

```
QUESTION: Does any file in this PR contain the word "workaround"?
```

```bash
grep -ri "workaround" --include="*.py" --include="*.md" .
```

| Result | Gate Result | Action |
|--------|-------------|--------|
| Found "workaround" | ❌ **REJECTED** | Replace with "fix", "resolution", or "mitigation". |
| No matches | ✅ **PASS** | All gates passed. 🚀 |

---

## FINAL CHECKLIST

Before clicking "Create Pull Request", verify ALL of these:

- [ ] **G1**: This is a FIX, not a warning/log/print
- [ ] **G2**: The PR is actually open on the upstream repo (URL: _________)
- [ ] **G3**: Reproduction works on CPU or free GPU (Colab/Kaggle)
- [ ] **G4**: PR description cites NeuralDBG (link + detection evidence)
- [ ] **G5**: Follow-up plan defined (comment after __ days on _______)
- [ ] **G6**: Zero occurrences of the word "workaround"

**If any box is unchecked, DO NOT create the PR. Fix the issue first.**

---

## Post-PR Timeline

| Day | Action |
|-----|--------|
| Day 0 | PR submitted. Share link on X, Discord, PyTorch Forums. |
| Day 3 | If no review: add a comment with additional test results or a simpler repro. |
| Day 7 | If no review: polite ping. "@maintainer, is there anything I can add to help review?" |
| Day 14 | If no review: escalate. Post on PyTorch Dev Discussions, ask for guidance. |
| Day 30 | If still no review: document in PLAN.md, move to next bug. Re-ping monthly. |
