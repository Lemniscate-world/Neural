# RULE 89: Lessons Learned - Rule Creation from Problems Solved

## Rule

When a problem is encountered and solved (bug, security issue, workflow error, misconfiguration), the solution MUST be captured as a rule to prevent recurrence.

## Rationale

AI agents and projects often repeat the same mistakes. By systematically converting solved problems into rules, we build institutional knowledge that improves all future work.

## Trigger Conditions

Create/Update a rule when ALL conditions are met:
1. **Error/Problem occurred** (bug, security issue, workflow error, misconfiguration)
2. **Root cause identified** and solution implemented
3. **A rule can prevent recurrence**

If error is one-off or too specific: document in project's LESSONS.md instead.

## Process

### Step 1: Problem Detection
- During development, testing, security scans, or code review
- Document what went wrong

### Step 2: Solution Capture
- Write a 1-2 sentence rule in appropriate location:
  - **kuro-rules/rules/** for cross-project rules
  - **AGENTS.md** for project-specific rules
  - **docs/architecture/** for technical decisions

### Step 3: Verification
- Run the rule against existing code
- If rule violation exists, fix it immediately

## Minimum Rule Template

```
# RULE XX: [Descriptive Title]

## Problem
[What went wrong]

## Solution
[How to prevent it]

## Verification
[How to check it doesn't happen again]
```

## Example

**Problem**: SESSION_SUMMARY.md and bandit*.json were committed to git (should be in .gitignore)

**Solution**: Added verification step in R76

```markdown
# RULE 76: .gitignore Security Patterns
...
Verification:
2. git ls-files | grep -E "SESSION_SUMMARY|bandit|safety-report|trivy" (should return nothing)
3. Before every PR: git status to verify no protected files are staged
```

## Enforcement

```
IF error occurs AND root cause identified AND rule can prevent recurrence:
  ACTION: Create or update a rule before session ends
  IF no appropriate file exists: create new rule in kuro-rules/rules/
  IF rule affects multiple projects: add to AGENTS.md index
  IF rule is project-specific: add to project AGENTS.md

IF error is one-off or too specific:
  ACTION: Document in project's LESSONS.md
```

## Project-Specific Learning

Each project should maintain a **LESSONS.md** file that captures:
- Problems encountered
- Solution implemented
- Rule created (if any)

---

**Created**: 2026-05-11
**Trigger**: SESSION_SUMMARY.md and bandit files were tracked by mistake
**Enforcement**: RECOMMENDED (but strongly encouraged)