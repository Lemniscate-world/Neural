# RULE 14.5: 5-Risk Failure Mode Table - Full Detail

## Trigger: Before first line of production code on any new project.

## The 5 Mandatory Risk Categories
| # | Category | Question to answer |
|---|----------|-------------------|
| 1 | Market Risk | Does the problem exist at scale? Is it painful enough to pay for? |
| 2 | Competition Risk | Is there an incumbent with distribution advantage we cannot overcome? |
| 3 | Technical Risk | Can we actually build this with our current stack and skills? |
| 4 | Regulatory Risk | Are there legal/compliance barriers? GDPR, financial regulations, etc.? |
| 5 | Adoption Risk | Even if we build it, will users change their behavior to use it? |

## Required Format per Risk
Risk: [Name]
Probability: Low / Medium / High
Impact: Low / Medium / High
Evidence FOR this risk: [specific data point or signal]
Evidence AGAINST this risk: [specific data point or signal]
Remedy: [concrete mitigation action]
Kill condition: [at what point do we stop the project over this risk?]

## Scoring
- 3+ HIGH probability + HIGH impact risks -> STOP, redesign or pivot
- 1-2 HIGH risks with clear remedies -> proceed with caution, monitor weekly
- All LOW/MEDIUM -> proceed

## Document Location
Save in: docs/failure_mode_table.md (add to .gitignore if sensitive)

## Enforcement
IF project starts without this table:
  ACTION: STOP, create table before first commit
  DO NOT: Treat this as optional documentation
# RULE 14: Periodic Validation - Full Detail
# RULE 73: Session Continuity Check

---

# RULE 14: Periodic Validation

## Applies to: CEO/Product persona only. DevOps/MLOps: see R28.

## Gates by Milestone

### 25% - Problem-Solution Fit
Mom Test (3+ new users, startup only): confirm problem still exists.
Marketing Test: landing page live, 50+ visitors, >= 5% email signup conversion.
Pass: document "VALIDATION_PASSED - 25%" in SESSION_SUMMARY.md

### 50% - Solution-Market Fit
Mom Test (5+ new users): would they pay? what is missing?
Marketing Test: 100+ visitors, 10+ qualified signups, 1+ beta tester willing.

### 75% - Pre-Launch Readiness
Mom Test (different user segments - not same people as before).
Marketing Test: pricing page tested, 1+ pre-order or letter of intent.

### 90% - Launch Readiness
5+ users completed the full user journey.
0 critical bugs in primary flow.
Support/feedback channel active.

### 95% - Final Gate
All above complete. No open critical issues.
Write "VALIDATION_PASSED - 95%" in SESSION_SUMMARY.md.

## Hard Lock (R20)
IF progress >= milestone AND "VALIDATION_PASSED - X%" NOT in SESSION_SUMMARY.md:
  SYSTEM LOCK - no code changes permitted.
  User must provide evidence and write VALIDATION_PASSED line manually.

## Enforcement
STOP at each gate. No exceptions for "we are so close."
Document in SESSION_SUMMARY.md, not just in chat.

---

# RULE 73: Session Resume Protocol

## Trigger: user says "on continue", "let us resume", "where were we"

## Required Actions (in order)
1. Read SESSION_SUMMARY.md - most recent entry first. Extract: last progress %, last branch, last task, next steps, blockers.
2. Check Linear - pull open CEO tasks and DevOps tasks.
3. Verify branch - confirm NOT on main. If on main: create branch first (R30).
4. Report status - R36 format (FR + EN).
5. Confirm with user: "Voici ou on en est. On continue avec [next step]?"

## Format
Resume - YYYY-MM-DD
Francais:
Branche: [branch]
Derniere progression: X%
Derniere action: [what was done]
Prochaine etape: [what to do next]
Blockers: [if any]

## Enforcement
IF session starts and project context exists:
  DO NOT: Start working without reading SESSION_SUMMARY.md first
  DO NOT: Ask user to remind you where you left off - read the file
  DO: Provide status report before any action
