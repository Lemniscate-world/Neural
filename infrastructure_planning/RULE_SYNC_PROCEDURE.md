# Rule Synchronization Procedure - NeuralDBG

## Overview

This document describes the complete procedure for synchronizing AI rules across all branches and team members.

## The Problem

As CEO, when you make structural changes or rule updates:
1. All team members must receive these changes immediately
2. DevOps must always see your tasks and progress in Linear
3. Forgetting to sync rules across branches causes chaos

## The Solution - 4 Layers of Defense

### Layer 1: GitHub Actions (Automatic)

**File:** `.github/workflows/rule-sync.yml`

When you push to a `ceo/` branch:
- Automatically detects rule file changes
- Syncs rules to `main` and `infra/milestone-0-setup`
- Creates a PR for review
- Updates `kuro-rules` master repository

**Trigger:** Push to any `ceo/**` branch

### Layer 2: Pull Request Template (Mandatory)

**File:** `.github/PR_TEMPLATES/default.md`

Every PR must complete the Rule 33 checklist:
- Verify branch is correct (`ceo/` for rules)
- Confirm rules are synced
- Check security scans pass
- Confirm test coverage

### Layer 3: Manual Sync Script (Emergency)

**File:** `scripts/sync_rules.py`

For urgent syncs or when automation fails:

```bash
# Sync to specific branch
python scripts/sync_rules.py --branch main

# Sync to all branches
python scripts/sync_rules.py --all-branches

# Sync to kuro-rules only
python scripts/sync_rules.py --kuro-rules-only

# Verify consistency
python scripts/sync_rules.py --verify
```

### Layer 4: AI Session Verification (Automated)

At every AI session start:
1. Agent reads AGENTS.md
2. Agent checks current branch
3. Agent verifies rules are current
4. Agent reports Rule 33 status

## Branch Strategy

```
ceo/kuro-semantic-event-structures  <- CEO rule changes ONLY
infra/milestone-0-setup            <- DevOps work
main                                <- Stable code
```

### Rule Authority

| Branch Scope | Can Modify Rules? | Can Modify Code? |
|--------------|-------------------|------------------|
| `ceo/**`     | YES               | YES              |
| `infra/**`   | NO (merge only)   | YES              |
| `feat/**`    | NO (merge only)   | YES              |
| `fix/**`     | NO (merge only)   | YES              |
| `main`       | NO (merge only)   | NO (protected)   |

## Linear Integration

### For CEO (You)

1. Create tasks in Linear for rule changes
2. Use label: `CEO Decision`
3. Track progress in Linear
4. AI agent checks Linear at session start

### For DevOps

1. All tasks visible in Linear
2. Labels: `DevOps`, `MLOps`, `Needs Review`
3. AI agent acts as reviewer for PRs
4. Progress tracked in Linear

## Workflow Summary

### When You Update Rules (as CEO)

1. **Push to `ceo/` branch**
   - Workflow auto-triggers
   - Rules sync to all branches
   - PR created automatically

2. **Merge PR**
   - Rules now on main
   - Team pulls latest
   - Done

### When DevOps Works

1. **Check Linear** for tasks
2. **Pull latest** from main (includes rules)
3. **Create branch** from main: `infra/MLO-X-description`
4. **Work and test**
5. **Create PR** with template
6. **AI reviews** (Rule 28)

### When AI Agent Starts

1. **Reads AGENTS.md** (Rule 1)
2. **Checks branch** - must not be main
3. **Verifies rules** - up to date?
4. **Reports status** - Rule 33 verification
5. **Checks Linear** - any new tasks?

## Emergency Procedures

### GitHub Actions Fails

```bash
# Manual sync
python scripts/sync_rules.py --all-branches
```

### Rules Out of Sync

```bash
# Verify consistency
python scripts/sync_rules.py --verify
```

### New Team Member

1. Clone repo
2. AI agent runs onboarding (Rule 32)
3. Agent creates first branch
4. Agent assigns Linear task
5. Done

## Files Modified

| File | Purpose |
|------|---------|
| `.github/workflows/rule-sync.yml` | Automatic sync on CEO push |
| `.github/PR_TEMPLATES/default.md` | Mandatory PR checklist |
| `scripts/sync_rules.py` | Manual sync script |
| `AGENTS.md` | Core rules (Rule 33) |
| `infrastructure_planning/RULE_SYNC_PROCEDURE.md` | This document |

## Key Rules Summary

| Rule | Description |
|------|-------------|
| Rule 11 | Sync to kuro-rules on update |
| Rule 15 | Sync all rule files together |
| Rule 28 | AI reviews DevOps PRs |
| Rule 30 | Always use branches |
| Rule 33 | Global rule parity |

## Checklist for Each Session

**For AI Agent:**
- [ ] Read AGENTS.md
- [ ] Check current branch
- [ ] Verify rules up to date
- [ ] Check Linear for tasks
- [ ] Report Rule 33 status

**For CEO (You):**
- [ ] Make rule changes on `ceo/` branch
- [ ] Push - automation handles sync
- [ ] Verify PR created
- [ ] Track in Linear

**For DevOps:**
- [ ] Pull latest from main
- [ ] Create branch from main
- [ ] Complete PR checklist
- [ ] Wait for AI review

## Support

If something breaks:
1. Check GitHub Actions logs
2. Run `python scripts/sync_rules.py --verify`
3. Check Linear for blocked tasks
4. Ask AI agent: "What does Rule 33 say?"

---

**Last Updated:** 2026-03-03
**Version:** 1.0.0
