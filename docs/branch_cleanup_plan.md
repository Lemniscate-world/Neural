# Branch Cleanup Plan - NeuralDBG

Date: 2026-05-29
Tracking issue: #661

## Current Branch Inventory

| Branch | Upstream | Status | Decision |
|---|---|---|---|
| `main` | `origin/main` | Clean, tracks remote main | Keep |
| `fix/658-fallback-hardening` | `origin/fix/658-fallback-hardening` | Active PR #662 | Keep until merged, then delete local and remote branch |
| `sec/NDBG-precommit-hardening` | none | Local-only, unmerged, risky test commit | Do not push; candidate for delete after confirmation |

## Risk Review

`sec/NDBG-precommit-hardening` contains one commit, `1309f6d0`, named `test: tentative de commit fichier sensible`.
Compared to `main`, it deletes 487 lines across security scripts, rules, tests, and CI files, and adds `test_commit_error.md`.
This looks like a guard-test branch, not production work.

## Recommended Cleanup

1. Finish PR #662.
2. After merge, delete the merged branch:

```powershell
git checkout main
git pull --ff-only
git branch -d fix/658-fallback-hardening
git push origin --delete fix/658-fallback-hardening
```

3. For `sec/NDBG-precommit-hardening`, first preserve evidence if needed:

```powershell
git format-patch main..sec/NDBG-precommit-hardening -o outputs/branch-archive/sec-NDBG-precommit-hardening
```

4. Then delete only after explicit confirmation:

```powershell
git branch -D sec/NDBG-precommit-hardening
```

## Branch Hygiene Policy

- No local branch should remain without an upstream unless it is less than 24 hours old.
- Any active branch must map to a GitHub issue.
- Test branches that intentionally trigger security guards must never be pushed.
- Before cleanup, run:

```powershell
git fetch --all --prune
git branch -vv --all
git branch --merged main
git branch --no-merged main
```
