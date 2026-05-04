# AGENTS.md â€” Strict Rules for AI Agents

**Purpose**: This file contains STRICT, ENFORCEABLE rules. Unlike `AI_GUIDELINES.md` (philosophy and best practices), this file is a CONTRACT that AI agents MUST follow. Violations MUST be reported to the user.

**Sync**: This file MUST be synced to all projects. When updating, sync to `~/Documents/kuro-rules` (master copy).

---

## RULE 1: Read Rules First â€” MANDATORY

### Rule
AI agents MUST read this file at the START of every session, BEFORE any other action.

### Verification
```
ACTION: Read AGENTS.md (this file) first
VERIFY: Confirm to user "I have read AGENTS.md and will enforce all rules"
```

### Enforcement
IF agent starts working without reading rules:
- STOP immediately
- READ AGENTS.md
- RESTART the task with rules in context

---

## RULE 2: Mom Test Gate â€” MANDATORY

### Rule
Project CANNOT exceed 10% progress until Mom Test is COMPLETE. No production code allowed during Mom Test phase.

### Verification Checklist
Before ANY code implementation, VERIFY ALL of the following:

| Requirement | Verification Method |
|-------------|---------------------|
| Minimum 5 interviews | Check `mom_test_results.md` has 5+ interview entries |
| `mom_test_script.md` exists | File exists with EN/FR interview questions |
| `mom_test_results.md` exists | File exists with documented interviews |
| `decision.md` exists | File exists with Go/No-Go/Pivot justification |
| 3+ spontaneous mentions | Count in `mom_test_results.md` |
| 2+ solution seekers | Count in `mom_test_results.md` |

### Enforcement
```
IF any checklist item is FALSE:
  ACTION: STOP implementation immediately
  ACTION: SET progress to 10% maximum
  ACTION: REPORT missing deliverables to user
  ACTION: COMPLETE missing deliverables before proceeding
  DO NOT: Write production code
  DO NOT: Create architecture documents beyond brainstorming
```

### Allowed During Mom Test (0-10%)
- Collecting interviews
- Creating `mom_test_script.md`
- Documenting in `mom_test_results.md`
- Creating `decision.md`
- Brainstorming in `ideas.md` (NO production code)
- Discussing approaches with user

### Forbidden During Mom Test (0-10%)
- Writing production code
- Implementing features
- Creating architecture beyond high-level brainstorming
- Setting progress above 10%

---

## RULE 3: Progress Tracking â€” MANDATORY

### Rule
Every project MUST track progress in `SESSION_SUMMARY.md` with PESSIMISTIC estimates.

### Progress Calculation

| Component | Weight | When Complete |
|-----------|--------|---------------|
| Mom Test | 10% | All deliverables done, decision made |
| Core functionality | 40% | All features working and tested |
| Test coverage (60%+) | 20% | Coverage report shows 60%+ |
| Security hardening | 10% | All scans pass (bandit, safety, etc.) |
| CI/CD & DevOps | 10% | Pipeline configured and passing |
| Documentation | 10% | README, CHANGELOG, API docs complete |

### Verification
```
BEFORE reporting progress:
  CALCULATE: Sum of completed components
  SUBTRACT: 10-15% for optimism bias
  VERIFY: Does this match reality?
  IF doubt: Subtract another 10%
```

### Enforcement
```
IF progress > actual completion:
  ACTION: Recalculate with pessimistic estimate
  ACTION: Document what's missing
  DO NOT: Inflate progress to make user happy
```

---

## RULE 4: Session Summary â€” MANDATORY

### Rule
Every session MUST update `SESSION_SUMMARY.md` with BOTH English and French versions.

### Required Format
```markdown
# Session Summary â€” YYYY-MM-DD
**Editor**: (VS Code | Cursor | Antigravity | Windsurf)

## Francais
**Ce qui a ete fait** : (Liste)
**Initiatives donnees** : (Nouvelles idees/directions)
**Fichiers modifies** : (Liste)
**Etapes suivantes** : (Ce qu'il reste a faire)

## English
**What was done**: (List)
**Initiatives given**: (New ideas/directions)
**Files changed**: (List)
**Next steps**: (What's next)

**Tests**: X passing
**Blockers**: (If any)
**Progress**: X% (pessimistic estimate)
```

### Verification
```
AT END of session:
  CHECK: SESSION_SUMMARY.md updated?
  CHECK: Both EN and FR sections present?
  CHECK: Progress percentage included?
  IF missing: CREATE/UPDATE before ending
```

### Enforcement
```
IF session ends without summary:
  ACTION: Create summary immediately
  ACTION: Include all required sections
  DO NOT: Skip this step
```

---

## RULE 5: Testing Requirements â€” MANDATORY

### Rule
All code MUST have minimum 60% test coverage. No exceptions.

### Testing Pyramid
| Type | Percentage | Purpose |
|------|---------|---------|
| Unit Tests | 70% | Test individual functions/methods |
| Integration Tests | 20% | Test component interactions |
| E2E Tests | 10% | Test complete user flows |

### Verification
```
BEFORE declaring feature complete:
  RUN: pytest --cov (or equivalent)
  CHECK: Coverage >= 60%?
  CHECK: All tests passing?
  IF fail: WRITE more tests
```

### Enforcement
```
IF coverage < 60%:
  ACTION: STOP new feature development
  ACTION: WRITE tests until 60%+ coverage
  DO NOT: Merge code without tests
```

---

## RULE 6: Security Scanning â€” MANDATORY

### Rule
All code MUST pass security scans before commit.

### Required Scans by Language

| Language | Commands |
|----------|----------|
| Python | `bandit -r .` AND `safety check` |
| Rust | `cargo audit` AND `cargo clippy` |
| Node.js/React | `npm audit` |

### Verification
```
BEFORE commit:
  RUN: Appropriate security scanner for language
  CHECK: All issues resolved?
  IF issues found: FIX before committing
```

### Enforcement
```
IF security scan fails:
  ACTION: STOP commit
  ACTION: FIX security issues
  ACTION: RE-RUN scan
  DO NOT: Commit with security vulnerabilities
```

---

## RULE 7: No Silent Failures â€” MANDATORY

### Rule
If any step fails, the agent MUST report it and retry. Never ignore failures.

### Verification
```
AFTER every tool use:
  CHECK: Did it succeed?
  IF failed:
    REPORT: Tell user what failed and why
    RETRY: Attempt the action again
    IF still failing: ASK user for help
  DO NOT: Continue as if nothing happened
```

### Enforcement
```
IF agent ignores a failure:
  THIS IS A RULE VIOLATION
  User should report: "Did you follow AGENTS.md?"
  Agent must: Acknowledge and fix the issue
```

---

## RULE 8: Critical Thinking â€” MANDATORY

### Rule
AI agents are CO-ENGINEERS, not typists. Push back on bad ideas.

### Required Questions Before Implementation

1. **"Does this actually help users?"**
   - If NO: Push back, suggest alternatives

2. **"Is there a simpler way?"**
   - If YES: Propose the simpler solution

3. **"What breaks?"**
   - Identify edge cases and failure modes

### Verification
```
BEFORE implementing:
  ASK: All 3 questions above
  DOCUMENT: Answers in response
  IF concerns: VOICE them to user
```

### Enforcement
```
IF agent implements without questioning:
  THIS IS A RULE VIOLATION
  Agent should: Proactively identify issues
  User can ask: "Did you apply critical thinking?"
```

---

## RULE 9: No Emojis Anywhere â€” MANDATORY

### Rule
Emojis are FORBIDDEN in ALL project files, code, comments, documentation, CLI output, and user-facing text. No exceptions.

### Reason
- Encoding issues across platforms
- Break compatibility with certain tools and terminals
- Reduce professionalism
- Distract from content

### Verification
```
BEFORE any output:
  CHECK: Does this contain emojis?
  IF YES: REMOVE all emojis
  CHECK: Does code/comments contain emojis?
  IF YES: REMOVE them
```

### Enforcement
```
IF emoji found in any file:
  ACTION: REMOVE immediately
  ACTION: WARN user if emoji was in user-provided content
  DO NOT: Add emojis to any output
```

---

## RULE 10: File Protection â€” MANDATORY

### Rule
Certain files MUST be in `.gitignore` and NEVER committed publicly.

### Protected Files
| File | Reason |
|------|--------|
| `mom_test_results.md` | Private interview data |
| `ideas.md` | Work-in-progress brainstorms |
| `architecture_notes.md` | Work-in-progress architecture |
| `concept/` | Strategy and vision folder |
| `mom_test_script.md` | Interview questions |
| `decision.md` | Strategic decisions |
| `.env` | Secrets and credentials |
| API keys, tokens | Security |

### Verification
```
BEFORE commit:
  CHECK: Are protected files in .gitignore?
  CHECK: Are any protected files being committed?
  IF protected file in commit: REMOVE from commit
```

### Enforcement
```
IF protected file is committed:
  ACTION: REMOVE from git history
  ACTION: ADD to .gitignore
  ACTION: WARN user about exposure
```

---

## RULE 11: Sync Rule â€” MANDATORY

### Rule
When rules are updated in ANY project, SYNC to `~/Documents/kuro-rules` (master copy).

### Verification
```
WHEN updating rules:
  CHECK: Is this update in kuro-rules?
  IF NO: COPY update to kuro-rules
  CHECK: Are other projects using old rules?
  IF YES: SYNC new rules to those projects
```

### Enforcement
```
IF rules are updated without sync:
  ACTION: SYNC to kuro-rules immediately
  ACTION: Update all affected projects
```

---

## RULE 12: Roadmap Adherence â€” MANDATORY

### Rule
Every project MUST have a roadmap file (PLAN.md or ROADMAP.md) and all development MUST follow it.

### Verification
```
BEFORE starting any task:
  CHECK: Does PLAN.md or ROADMAP.md exist?
  CHECK: Is the task aligned with the roadmap?
  IF NO roadmap: CREATE one before coding
  IF task NOT in roadmap: ASK user for confirmation
```

### Roadmap Requirements
- Clear build order with numbered steps
- Success criteria for each phase
- Anti-goals (what NOT to build)
- MVP scope definition

### Enforcement
```
IF no roadmap exists:
  ACTION: STOP and create PLAN.md
  ACTION: Define MVP scope, build order, success criteria
  DO NOT: Write code without a plan

IF code deviates from roadmap:
  ACTION: ASK user if roadmap should be updated
  ACTION: Document the deviation reason
  DO NOT: Silently ignore the plan
```

### Progress Alignment
- Roadmap phases should map to progress percentages
- Each completed phase updates SESSION_SUMMARY.md progress
- Roadmap changes require explicit user approval

---

## RULE 13: Roadmap Duration â€” MANDATORY

### Rule
Every roadmap MUST have a minimum duration of **one month** with clearly defined phases.

### Verification
```
BEFORE creating PLAN.md:
  CHECK: Does the roadmap span at least 4 weeks?
  CHECK: Are phases clearly defined with start/end dates?
  CHECK: Is there a realistic scope for each phase?
  IF duration < 1 month: EXPAND scope or EXTEND timeline
```

### Roadmap Duration Requirements
- Minimum 4 weeks of planned work
- Weekly milestones or checkpoints
- Clear deliverables for each phase
- Buffer time for unexpected issues (10-15%)

### Progress Calculation Integration
The roadmap progress contributes to overall SESSION_SUMMARY.md progress:

| Component | Weight | Calculation |
|-----------|--------|-------------|
| Roadmap Phase Completion | Sub-component of Core Functionality | (Completed Phases / Total Phases) Ã— 40% |
| Phase Quality | Multiplier | 0.5x (incomplete) to 1.0x (fully tested) |

### Enforcement
```
IF roadmap duration < 1 month:
  ACTION: STOP and expand the plan
  ACTION: Add more phases or extend timeline
  DO NOT: Start coding with insufficient planning horizon
```

---

## RULE 14: Periodic Validation â€” MANDATORY

### Rule
At progress milestones (25%, 50%, 75%, 90%, 95%), the product MUST be validated through Mom Test and Marketing Test before continuing.

### Validation Gates

| Progress Milestone | Required Validation |
|-------------------|-------------------|
| 25% | Mom Test follow-up (3+ users), Marketing Test (landing page views, signups) |
| 50% | Mom Test validation (5+ new users), Marketing Test (conversion metrics) |
| 75% | Mom Test expansion (different user segments), Marketing Test (pricing validation) |
| 90% | Final Mom Test (comprehensive), Marketing Test (launch readiness) |
| 95% | Pre-launch validation (all criteria met) |

### Validation Checklist
```
AT each milestone:
  CHECK: Mom Test conducted with new users?
  CHECK: Marketing Test metrics collected?
  CHECK: User feedback documented?
  CHECK: Pivot/continue decision made?
  IF validation FAILED:
    ACTION: STOP development
    ACTION: Address feedback or pivot
    DO NOT: Continue without validation
```

### Mom Test Requirements
- Interview minimum 3-5 new users at each milestone
- Ask about actual behavior, not opinions
- Document spontaneous mentions and solution-seeking behavior
- Update `mom_test_results.md` with new findings

### Marketing Test Requirements
- Landing page or demo available
- Track views, signups, engagement
- Document conversion metrics
- Validate pricing hypothesis (if applicable)

### Enforcement
```
IF milestone reached without validation:
  ACTION: STOP immediately
  ACTION: Conduct validation before continuing
  DO NOT: Skip validation gates
```

---

## RULE 15: Rule Synchronization â€” MANDATORY

### Rule
When ANY rule file is updated, ALL rule files MUST be updated to include the same rule. Rules must be consistent across AGENTS.md, AI_GUIDELINES.md, .cursorrules, copilot-instructions.md, and GAD.md.

### Verification
```
AFTER updating any rule file:
  CHECK: Is this rule in all other rule files?
  IF NO: ADD the rule to all files
  CHECK: Is wording consistent?
  IF NO: SYNC wording across files
```

### Enforcement
```
IF rules are inconsistent across files:
  ACTION: SYNC immediately to all files
  ACTION: Document sync in SYNC_LOG.md
  DO NOT: Allow rule drift between files
```

---

## RULE 16: Working Demos â€” MANDATORY

### Rule
At each validation milestone (25%, 50%, 75%, 90%, 95%), the project MUST have at least **2 working demos** that demonstrate core functionality.

### Requirements
- Minimum 2 demos per milestone
- Each demo must be runnable without errors
- Demos must demonstrate different aspects of the product
- Demos must be documented with expected output

### Verification
```
AT each milestone:
  CHECK: Are there at least 2 demos?
  CHECK: Do all demos run successfully?
  CHECK: Do demos demonstrate different features?
  IF demos < 2:
    ACTION: STOP and create missing demos
    DO NOT: Continue without 2 working demos
```

### Enforcement
```
IF milestone reached without 2 working demos:
  ACTION: STOP immediately
  ACTION: Create/fix demos until 2 are working
  DO NOT: Skip this requirement
```

---

## RULE 17: Deep Understanding Before Phase Transition â€” MANDATORY

### Rule
Before transitioning to the next phase, the user MUST demonstrate deep understanding of what was created, including 2nd and 3rd order consequences.

### Requirements
1. **Explain the mechanism**: How does it work under the hood?
2. **2nd order consequences**: What happens if this is used in production? What edge cases emerge?
3. **3rd order consequences**: What long-term effects? What dependencies form?
4. **Teach something new**: Agent must teach user at least one new concept
5. **Critical thinking prompts**: Agent must ask probing questions about the creation

### Verification Checklist
```
BEFORE phase transition:
  CHECK: Can user explain the mechanism?
  CHECK: Have 2nd/3rd order consequences been discussed?
  CHECK: Has user learned something new?
  CHECK: Have critical thinking questions been asked?
  IF NOT:
    ACTION: STOP and provide deep explanation
    ACTION: Ask probing questions
    ACTION: Teach new concepts
    DO NOT: Transition without understanding
```

### Critical Thinking Questions (Agent MUST Ask)
1. "What could break this in production that we haven't tested?"
2. "What would happen if 10x more users used this?"
3. "What assumptions are we making that might be wrong?"
4. "What would you do if this completely failed?"
5. "What did you learn that surprised you?"

### Enforcement
```
IF phase transition requested without deep understanding:
  ACTION: STOP and provide explanation
  ACTION: Ask all 5 critical thinking questions
  ACTION: Discuss 2nd and 3rd order consequences
  DO NOT: Allow superficial understanding
```

---

## RULE 18: Regression Prevention â€” MANDATORY

### Rule
A **regression** is a bug that appears in a previously functional feature after a code change. AI agents MUST prevent regressions by verifying the entire system state after any modification.

### Verification Checklist
```
AFTER any change (fix, feature, or refactor):
  1. RUN: Entire test suite (not just the local module)
  2. CHECK: Did previously passing tests fail?
  3. VERIFY: Mocks match production data structures exactly
  4. ENSURE: Fake timers are isolated and cleaned up
  5. CONFIRM: No "null" returns in mocks when objects/arrays are expected
```

### Enforcement
```
IF a regression is detected:
  ACTION: STOP new work
  ACTION: FIX the regression immediately
  ACTION: DOCUMENT why it happened (mock mismatch, side effect, etc.)
  DO NOT: Ignore failing tests from "unrelated" modules
```

---

## RULE 19: Strict Versioning â€” MANDATORY

### Rule
Every project MUST follow Semantic Versioning (SemVer) with author attribution (e.g., `v0.1.0-kuro`). Stable releases MUST be tagged at each validation milestone.

### Verification Checklist
```
AT each validation milestone (25%, 50%, 75%, 90%, 95%):
  1. VERIFY: Code is stable and entire test suite passes
  2. GENERATE: Release tag with SemVer + Author (e.g. v0.1.0-kuro)
  3. PUSH: Tag to repository
```

### Enforcement
```
IF milestone reached without version tag:
  ACTION: STOP development
  ACTION: Create and push the version tag immediately
  DO NOT: Continue to next phase without a stable versioned release
```

---

## RULE 20: Hard Milestone Lock â€” CRITICAL

### Rule
STOP ALL code/system modifications if a progress milestone (Rule 14) is crossed without "VALIDATION_PASSED" in SESSION_SUMMARY.md. This is a hard lock.

### Verification Checklist
```
AT each validation milestone (25%, 50%, 75%, 90%, 95%):
  1. CHECK: Is "VALIDATION_PASSED" explicitly stated in SESSION_SUMMARY.md for the current milestone?
  2. IF NO: Trigger Hard Milestone Lock.
```

### Enforcement
```
IF a milestone is reached and "VALIDATION_PASSED" is NOT found in SESSION_SUMMARY.md:
  ACTION: SYSTEM LOCK - No code edits or system modifications are permitted.
  ACTION: User MUST provide validation results and explicitly state "VALIDATION_PASSED" in SESSION_SUMMARY.md.
  DO NOT: Proceed with any development until the lock is released.
```

---

---

## RULE 21: Intelligence Harvester â€” MANDATORY

### Rule
The agent MUST perform external market intelligence research at every milestone (10%, 25%, 50%, 75%, 90%, 95%). This involves searching at least 3 distinct sources (Reddit, App Store, specialized forums, etc.) to identify user pain points, competitor weaknesses, and market gaps.

### Verification Checklist
```
AT each milestone:
  1. SEARCH: At least 3 external sources for the project domain
  2. ANALYZE: Identify 2+ major user complaints about competitors
  3. SYNTHESIZE: Document how the current project addresses these "pain points"
  4. RECORD: Add the "Intelligence Report" to the milestone validation documentation
```

### Enforcement
```
IF milestone reached without Intelligence Report:
  ACTION: STOP development
  ACTION: Conduct and document the intelligence research immediately
  DO NOT: Continue implementation until market gaps are documented
```

---

---

## RULE 22: Feature Focus Rule â€” MANDATORY

### Rule
Development MUST focus on only ONE specific feature for each periodic validation cycle (25%, 50%, 75%, 90%, 95%). This focus on depth over breadth continues even after the MVP phase.

### Verification Checklist
```
AT each milestone:
  1. IDENTIFY: Which single feature is the focus of this validation cycle?
  2. VERIFY: Has this feature been implemented with maximum depth and robustness?
  3. CHECK: Are all other feature developments currently paused?
  4. CONFIRM: Is this rule being applied post-MVP?
```

### Enforcement
```
IF validation involves multiple shallow features or lacks a single focus:
  ACTION: STOP development
  ACTION: Re-focus on a single primary feature for this cycle
  ACTION: Ensure implementation depth meets standards before proceeding
  DO NOT: Sacrifice depth for breadth during validation
```

---

## RULE 23: Knowledge Capture â€” MANDATORY

### Rule
Every project failure or pivot MUST be documented in the central `kuro-rules/KNOWLEDGE_BASE/` to ensure cross-project intelligence and prevent repeating mistakes.

### Verification Checklist
```
AFTER a pivot or project termination:
  1. CREATE: A post-mortem document in `kuro-rules/KNOWLEDGE_BASE/`
  2. DOCUMENT: Rationale for failure/pivot and key technical or market learnings
  3. SYNC: Ensure this rule is added to all local project rule files
```

### Enforcement
```
IF a project pivots without a post-mortem:
  ACTION: STOP and document the failure in the master repository
  DO NOT: Start a new project without acknowledging previous learnings
```

---

## RULE 24: Marketing & Outreach Guardian â€” MANDATORY

### Rule
Before any public release or marketing campaign, the AI Agent MUST ensure proper community outreach, feedback collection channels, and communication templates are prepared.

### Requirements
1. **Community Identification**: Identify at least 3 relevant communities or forums where the product would be discussed.
2. **Feedback Channels**: Establish clear channels for user feedback such as Discord, GitHub Issues, or survey forms.
3. **Communication Templates**: Prepare outreach templates for announcements, bug reports, and feature requests.
4. **Launch Checklist**: Verify all marketing materials are reviewed for accuracy and current product scope.

### Verification Checklist
```
BEFORE launch or major announcement:
  CHECK: Are 3+ target communities identified?
  CHECK: Is feedback channel active and monitored?
  CHECK: Are communication templates drafted?
  CHECK: Has marketing content been reviewed?
  IF any missing:
    ACTION: STOP and prepare the missing marketing foundations
```

### Enforcement
```
IF launch or outreach begins without marketing foundations:
  ACTION: STOP the announcement or campaign
  ACTION: Prepare communities, feedback channels, and templates first
  DO NOT: Ship marketing without a response path
```

---

When asking "Did you follow AGENTS.md?", the agent MUST provide:

1.  **Rule 1**: "I read AGENTS.md at the start of this session"
2.  **Rule 2**: "Mom Test status: [COMPLETE/IN PROGRESS/NOT STARTED]"
3.  **Rule 3**: "Progress: X% (calculated as: [breakdown])"
4.  **Rule 4**: "SESSION_SUMMARY.md: [UPDATED/NEEDS UPDATE]"
5.  **Rule 5**: "Test coverage: X%"
6.  **Rule 6**: "Security scans: [PASSED/FAILED/PENDING]"
7.  **Rule 7**: "Any failures: [NONE/REPORTED]"
8.  **Rule 8**: "Critical thinking applied: [YES/NO - details]"
9.  **Rule 9**: "Emojis: [NONE FOUND/REMOVED]"
10. **Rule 10**: "Protected files: [SAFE/EXPOSED]"
11. **Rule 11**: "Rules synced: [YES/NO]"
12. **Rule 12**: "Roadmap: [EXISTS/MISSING] - Task aligned: [YES/NO]"
13. **Rule 13**: "Roadmap duration: [>=1 month/TOO SHORT]"
14. **Rule 14**: "Periodic validation: [DONE/PENDING/NOT REQUIRED YET]"
15. **Rule 15**: "All rule files synced: [YES/NO]"
16. **Rule 16**: "Working demos: [2+/1/0]"
17. **Rule 17**: "Deep understanding demonstrated: [YES/NO]"
18. **Rule 18**: "Regression prevention: [FOLLOWED - entire suite ran?]"
19. **Rule 19**: "Strict Versioning: [vX.Y.Z-author tag created?]"
20. **Rule 20**: "Hard Milestone Lock: [LOCKED/UNLOCKED]"
21. **Rule 21**: "Intelligence Harvester: At least 3 sources analyzed for the current milestone? [YES/NO]"
22. **Rule 22**: "Feature Focus Rule: Only one feature focused on for this validation cycle? [YES/NO]"
23. **Rule 23**: "Knowledge Capture: Post-mortem documented for pivot/failure? [YES/NO]"
24. **Rule 24**: "Marketing & Outreach Guardian: Communities identified, feedback path ready, and templates drafted? [YES/NO]"
25. **Rule 25**: "MLOps/DevOps Collaboration: Infra guidance adapted for this task? [YES/NO]"
26. **Rule 26**: "DevOps/MLOps Tasks: 5 tasks generated and created as Linear issues? [YES/NO]"
27. **Rule 27**: "Persona Adaptability: Adapted vocabulary/depth for user role? [YES/NO]"
28. **Rule 28**: "Linear Automation: DevOps tasks created and assigned? [YES/NO]"
29. **Rule 29**: "Linear Integration: Connection verified at session start? [YES/NO]"
30. **Rule 30**: "Branch Creation: Working branch created before coding? [YES/NO]"
31. **Rule 31**: "Codebase Context: Issues include context for contributors? [YES/NO]"
32. **Rule 32**: "Team Stack: All required tools configured? [YES/NO]"
33. **Rule 33**: "Rule Parity: Branch has current rule set? [YES/NO]"
34. **Rule 34**: "Strict Project Isolation: Scope limited to the active project only? [YES/NO]"
35. **Rule 35**: "CEO Progress Visibility: Tasks visible in Linear? [YES/NO]"
36. **Rule 36**: "Session Status: Report provided at session start? [YES/NO]"
37. **Rule 37**: "Real-Time Linear Sync: CEO activities automatically synced? [YES/NO]"
38. **Rule 38**: "Code Review: Review completed before continuation? [YES/NO]"
39. **Rule 39**: "Pre-Marketing Due Diligence: Perplexity and Grok research completed before marketing claims? [YES/NO]"
40. **Rule 40**: "CEO Complete Dashboard: All tasks visible in Linear automatically? [YES/NO]"
41. **Rule 41**: "Personal Quant Mode (PQPO): (Rule 2/14/21/24 Bypassed) [YES/NO]"
45. **Rule 45**: "PR Analysis & Improvement: PRs strictly analyzed and improved? [YES/NO]"
814: **Rule 47**: "History Preservation: SESSION_SUMMARY.md history preserved? [YES/NO]"

---

## ENFORCEMENT SUMMARY

| Rule | Consequence of Violation |
|------|--------------------------|
| Rule 1 (Read First) | STOP and read rules |
| Rule 2 (Mom Test) | STOP implementation, complete deliverables |
| Rule 3 (Progress) | Recalculate with pessimistic estimate |
| Rule 4 (Session Summary) | Create summary immediately |
| Rule 5 (Testing) | STOP features, write tests |
| Rule 6 (Security) | STOP commit, fix vulnerabilities |
| Rule 7 (No Silent Failures) | Report and retry |
| Rule 8 (Critical Thinking) | Apply questions retroactively |
| Rule 9 (No Emojis) | REMOVE emojis immediately |
| Rule 10 (File Protection) | Remove from git, add to .gitignore |
| Rule 11 (Sync) | Sync to kuro-rules immediately |
| Rule 12 (Roadmap) | STOP and create PLAN.md if missing |
| Rule 13 (Roadmap Duration) | STOP and expand plan if < 1 month |
| Rule 14 (Periodic Validation) | STOP and conduct validation at milestones |
| Rule 15 (Rule Synchronization) | SYNC all rule files immediately |
| Rule 16 (Working Demos) | STOP and create 2 working demos |
| Rule 17 (Deep Understanding) | STOP and provide deep explanation |
| Rule 18 (Regression Prevention) | STOP and fix immediately |
| Rule 19 (Strict Versioning) | STOP and create tag immediately |
| Rule 20 (Hard Milestone Lock) | SYSTEM LOCK: No code edits permitted until validation results are provided |
| Rule 21 (Intel Harvester) | STOP and conduct intelligence research immediately |
| Rule 22 (Feature Focus) | STOP and re-focus on a single feature |
| Rule 24 (Marketing & Outreach Guardian) | STOP and prepare communities, feedback paths, and templates |
| Rule 25 (MLOps/DevOps Collaboration) | Adjust guidance toward infrastructure, delivery, and reliability |
| Rule 26 (DevOps/MLOps Tasks) | STOP and generate 5 tasks |
| Rule 27 (Persona Adaptability) | Adjust communication style |
| Rule 28 (Linear Automation) | Create Linear issues |
| Rule 29 (Linear Integration) | STOP and configure Linear |
| Rule 30 (Branch Creation) | STOP and create branch |
| Rule 31 (Codebase Context) | Add context to issues |
| Rule 32 (Team Stack) | Verify tool setup |
| Rule 33 (Rule Parity) | Sync rules across branches |
| Rule 34 (Strict Project Isolation) | STOP and filter scope to the target project ONLY |
| Rule 35 (CEO Progress) | Verify Linear visibility |
| Rule 36 (Session Status) | STOP and provide report first |
| Rule 37 (Real-Time Sync) | STOP and sync immediately |
| Rule 38 (Code Review) | STOP - review required before continuation |
| Rule 39 (Pre-Marketing Due Diligence) | STOP marketing claims and complete desk research first |
| Rule 40 (CEO Complete Dashboard) | STOP and restore complete Linear visibility |
| Rule 41 (Personal Quant Mode - PQPO) | SYSTEM LOCK: Fail any of 10 Gates or Testing Funnel |
| Rule 45 (PR Analysis & Improvement) | STOP and perform mandatory improvement cycle |
| **All Rules** | Cease work immediately - Do NOT bypass |

## FINAL NOTE

These rules are NON-NEGOTIABLE. They exist to ensure:
- User problems are validated before building solutions
- Code quality meets professional standards
- Security is never compromised
- Progress is accurately tracked
- Knowledge persists across sessions

---

## RULE 25: MLOps/DevOps Collaboration â€” MANDATORY

### Rule
When interacting with a DevOps or MLOps engineer on this repository, the AI Agent MUST shift its focus to infrastructure, delivery, and reliability.

### Verification Checklist
```
WHEN working on infrastructure/deployment:
  1. FOCUS: Are we prioritizing reproducibility and clean pipelines?
  2. SECURITY: Are security tools (bandit, cargo audit) strictly enforced in the CI/CD configuration proposals?
  3. MLOPS: Are we tracking experiments and versioning data appropriately?
```

### Enforcement
```
IF providing MLOps/DevOps assistance:
  ACTION: Provide production-ready configurations (Dockerfiles, YAML).
  ACTION: Propose architecture adjustments synchronously for ML model changes.
  DO NOT: Provide brittle or untestable infrastructure code.
```

---

## RULE 26: DevOps/MLOps Milestone Task Generation â€” MANDATORY

### Rule
At every progress milestone (10%, 25%, 50%, 75%, 90%, 95%), the AI Agent MUST strictly analyze the repository's current state and propose exactly **5 concrete DevOps or MLOps tasks**.

### Requirements
1. **Analysis-Driven**: Tasks must be based on a strict analysis of the current codebase and its bottlenecks.
2. **Resource Estimation**: Each task MUST include a strict estimation of the time or resources it will save the team.
3. **Documentation**: These tasks MUST be documented in the infrastructure_planning/ folder in TWO Markdown files: an English version (milestone_X_tasks.md) and a French pedagogical version (milestone_X_tasks_fr.md).
4. **Actionable**: Tasks must be ready for a DevOps/MLOps engineer to pick up.
5. **Linear Integration**: Each task MUST also be created as a Linear issue in the appropriate DevOps/MLOps team, with full description, acceptance criteria, and ROI estimation.

### Enforcement
```
IF a milestone is reached:
  ACTION: Analyze repo for infrastructure/pipeline needs.
  ACTION: Generate 5 DevOps/MLOps tasks with Return on Investment (ROI) estimations.
  ACTION: Save to `infrastructure_planning/milestone_X_tasks.md`.
  DO NOT: Skip this operational planning step.
```

---

## RULE 27: Persona Adaptability â€” MANDATORY

### Rule
Before initiating significant work or generating explanations, the AI Agent MUST identify or ask "Who is interacting with me? (e.g., CEO, DevOps, MLOps, Fullstack Dev)". The AI MUST adapt its depth of explanation, vocabulary, and feature propositions accordingly.

### Requirements
1. **CEO/Product Persona**: Focus on "Why". Explain business value, Mom Test integration, user impact, KPIs, time-to-market. Keep technical details abstract (ASCII diagrams).
2. **DevOps/MLOps Persona**: Focus on "How (Infra)". Discuss CI/CD gates, reproducible pipelines, determinism, network latency, security layers.
3. **Developer Persona**: Focus on "How (Code)". Discuss architecture, modularity, algorithmic complexity, DRY, SOLID.
4. **Pedagogy Engine**: If the persona is learning, provide highly detailed ASCII diagrams and step-by-step decoding.

### Enforcement
```
IF the user's role is known or stated:
  ACTION: Adjust vocabulary and technical depth immediately.
  ACTION: Emphasize the rules most relevant to that persona.
  DO NOT: Speak to a CEO like a DevOps, or a DevOps like a CEO, unless pedagogical translation is requested.
```

When in doubt, ASK the user. Do not assume.

When in doubt, ASK the user. Do not assume.

---

## RULE 28: Linear Automation and DevOps Review â€” MANDATORY

### Rule
At every milestone, the AI Agent MUST automatically create the 5 DevOps/MLOps tasks as Linear issues (Rule 26), assign them to the designated DevOps/MLOps engineer, and continuously track their progress. The AI Agent MUST act as a reviewer when the engineer submits work.

### Requirements
1. **Automatic Issue Creation**: The 5 tasks generated by Rule 26 MUST be automatically created as Linear issues with full descriptions, acceptance criteria, and ROI estimations.
2. **Assignment**: Issues MUST be assigned to the DevOps/MLOps engineer (currently: penielteko02@gmail.com in Linear).
3. **Official Labels**: Every Linear issue MUST use labels from the following official list. Do NOT create ad-hoc labels.

| Label | Usage |
|-------|-------|
| DevOps | CI/CD, Docker, GitHub Actions, pipelines, deployment |
| MLOps | Experiment tracking, data versioning, model registry, DVC, MLflow |
| Core Engine | Core engine logic (neuraldbg.py, causal inference, semantic events) |
| Validation | Mom Tests, user interviews, market validation |
| Documentation | Guides, README, session summaries, CODEBASE_GUIDE |
| Security | Security scans, bandit, safety, vulnerability fixes (Rule 6) |
| Milestone Task | Infrastructure tasks generated by Rule 26 |
| Testing | Tests, coverage, pytest, test infrastructure (Rule 5) |
| Needs Review | Code review required per Rule 28 |
| CEO Decision | Strategic decisions requiring CEO/Lead input |
3. **Progress Tracking**: The AI Agent MUST check Linear issue statuses when resuming sessions and report task progress.
4. **Code Review Role**: When the DevOps/MLOps engineer submits work (PR, branch, or issue update), the AI Agent MUST review it as a senior DevOps/MLOps reviewer:
   - Verify the work meets the acceptance criteria in the Linear issue.
   - Check for security compliance (Rule 6), test coverage (Rule 5), and reproducibility.
   - Provide constructive, pedagogical feedback (Rule 27 Persona: DevOps/MLOps).
5. **Git Branch Creation**: The AI Agent MUST always create a dedicated git branch for the user before starting work on any task.

### Enforcement
`
IF a milestone is reached:
  ACTION: Create 5 Linear issues automatically (Rule 26).
  ACTION: Assign all issues to the DevOps/MLOps engineer.
  ACTION: Create a git branch for the current milestone work.
  DO NOT: Skip Linear issue creation or assignment.

IF the DevOps/MLOps engineer submits work:
  ACTION: Review against acceptance criteria.
  ACTION: Check security, tests, and reproducibility.
  ACTION: Provide feedback as a senior reviewer.
  DO NOT: Accept work that does not meet the documented criteria.
`

---

## RULE 29: Mandatory Linear Integration â€” CRITICAL

### Rule
Every team member and every AI Agent MUST have a working connection to Linear before starting any work session. This is non-negotiable. Without Linear, no task tracking occurs, and work is invisible to the team.

### Integration Methods (by environment)

| Environment | Required Integration |
|-------------|---------------------|
| VS Code | Linear extension from VS Code Marketplace |
| Cursor | Linear extension OR MCP server (linear-mcp-server) |
| Antigravity | MCP server (linear-mcp-server) |
| Windsurf | MCP server (linear-mcp-server) |
| GitHub Codespaces | Linear GitHub integration + MCP server |
| Terminal-only | Linear CLI or MCP server |

### Requirements
1. **Session Gate**: The AI Agent MUST verify Linear connectivity at the start of every session. If unavailable, guide the user through setup before proceeding.
2. **Human Onboarding**: When a new team member joins, the FIRST task is to configure their Linear connection. No code is written until Linear is operational.
3. **Issue Visibility**: All tasks, bugs, and features MUST be trackable in Linear. Work done outside Linear is considered undocumented and violates traceability (Rule 4).

### Enforcement
`
IF Linear connection is not configured:
  ACTION: STOP all work.
  ACTION: Guide user through Linear setup for their IDE/environment.
  DO NOT: Allow any development work without Linear tracking.

IF a new team member joins:
  ACTION: First task is Linear setup and verification.
  ACTION: Assign them a test issue to confirm the connection works.
  DO NOT: Skip this onboarding step.
`

---

## RULE 30: Mandatory Branch Creation â€” CRITICAL

### Rule
NOBODY works on main directly. Before any work begins, the AI Agent MUST create or verify a dedicated git branch for the contributor. Every contributor gets their own branch, named according to a strict convention.

### Branch Naming Convention
`
[scope]/[issue-id]-[short-description]
`

| Scope | Usage | Example |
|-------|-------|---------|
| ceo/ | Strategic Development & Rule Management (CEO Only) | ceo/kuro-semantic-event-structures |
| infra/ | Infrastructure / DevOps / MLOps | infra/milestone-0-setup |
| feat/ | New feature development | feat/MLO-1-ci-cd-pipeline |
| fix/ | Bug fix | fix/MLO-3-docker-volume-error |
| docs/ | Documentation only | docs/update-readme-badges |
| refactor/ | Code refactoring | refactor/modularize-training |

5. **Global Consistency**: For tasks that span multiple repositories (e.g., rule syncs, platform migrations), the branch name MUST be identical across all affected repositories.

### Requirements
1. **Session Gate**: At the start of every session, the AI Agent MUST check the current branch. If on main, create or switch to the appropriate working branch immediately.
2. **One Branch Per Task**: Each Linear issue or task MUST have its own branch. Do not mix unrelated changes.
3. **Merge via PR Only**: Branches are merged into main exclusively through Pull Requests. Direct pushes to main are forbidden.
4. **Branch for Every Contributor**: When a new team member starts, the AI Agent MUST create their first working branch before any code is written.

### Enforcement
`
IF contributor is on main and about to write code:
  ACTION: STOP immediately.
  ACTION: Create a branch following the naming convention.
  ACTION: Switch to the new branch before any edits.
  DO NOT: Allow any code changes on main.

IF a Linear issue exists for the task:
  ACTION: Use the Linear issue ID in the branch name (e.g., feat/MLO-1-ci-cd).
  DO NOT: Create unnamed or generic branches (e.g., dev, 	est, 	emp).
`

---

## RULE 31: Codebase Context in Linear Issues -- MANDATORY

### Rule
Every Linear issue assigned to a team member MUST include a "Codebase Context" section that explains the relevant files, their purpose, and how they connect to the task. The goal is that a contributor who has NEVER seen the repo can understand exactly what to do.

### Requirements
1. **File Map**: List every file the contributor will need to read or modify, with a one-line explanation of what it does.
2. **Architecture Briefing**: Explain how the files relate to each other and to the project core architecture (Hub and Spokes).
3. **Key Concepts**: Define any domain-specific terms (e.g., "vanishing gradients", "causal compression") in plain language.
4. **Entry Point**: Tell the contributor where to START reading the code (which file, which function).
5. **Codebase Guide**: Maintain a permanent `infrastructure_planning/CODEBASE_GUIDE.md` file that provides a high-level map of the entire repository for new contributors.

### Enforcement
```
IF creating a Linear issue for a team member:
  ACTION: Include a "Codebase Context" section with file map, architecture briefing, and key concepts.
  ACTION: Update `infrastructure_planning/CODEBASE_GUIDE.md` if new files are added.
  DO NOT: Assume the contributor knows the codebase.
  DO NOT: Create issues that reference files without explaining them.
```

---

## RULE 32: Mandatory Team Stack -- CRITICAL

### Rule
Every team member MUST use the following standardized stack. The AI Agent MUST verify compliance at session start and guide setup if any tool is missing.

### Official Stack

| Category | Tool | Purpose | Required |
|----------|------|---------|----------|
| **Project Management** | Linear | Issue tracking, sprints, milestones, labels | YES |
| **IDE (Primary)** | Cursor | AI-assisted coding with MCP and rules support | YES (or alternative below) |
| **IDE (Alternative)** | VS Code / Antigravity / Windsurf | Coding with AI extensions | YES (one of these) |
| **Version Control** | Git + GitHub | Source control, PRs, branch protection | YES |
| **AI Integration** | MCP Server (linear-mcp-server) | Linear access from IDE | YES (Rule 29) |
| **CI/CD** | GitHub Actions | Automated testing, security, deployment | YES (Rule 26 Task 1) |
| **Containerization** | Docker + docker-compose | Hermetic dev environments | RECOMMENDED |
| **Experiment Tracking** | MLflow or W&B | ML experiment logging | RECOMMENDED (MLOps) |
| **Data Versioning** | DVC | Large file versioning | RECOMMENDED (MLOps) |
| **Language** | Python 3.10+ | Core development language | YES |
| **ML Framework** | PyTorch | Deep learning framework | YES |
| **Testing** | pytest + pytest-cov | Unit tests with coverage | YES (Rule 5) |
| **Security** | bandit + safety | Static analysis and dependency audit | YES (Rule 6) |
| **Communication** | Linear comments + GitHub PRs | Async team communication | YES |

### Onboarding Checklist
When a new team member joins, the AI Agent MUST walk them through this checklist:
```
[ ] Git configured (name, email)
[ ] GitHub access to the repository
[ ] IDE installed (Cursor recommended)
[ ] Linear account created and connected (Rule 29)
[ ] MCP server configured (linear-mcp-server)
[ ] Python 3.10+ installed
[ ] Virtual environment created (.venv)
[ ] Dependencies installed (pip install -e .)
[ ] Tests passing locally (pytest tests/)
[ ] Demo running (python demo_vanishing_gradients.py)
[ ] CODEBASE_GUIDE.md read
[ ] First working branch created (Rule 30)
```

### Enforcement
```
IF a new team member joins:
  ACTION: Present the onboarding checklist above.
  ACTION: Do NOT proceed with code until all YES items are confirmed.
  DO NOT: Allow coding without Linear + IDE + Git configured.

IF a session starts:
  ACTION: Verify the contributor has the required stack.
  ACTION: If missing, guide setup before any work.
```

---

## RULE 33: Global Rule Parity and Mandatory Cross-Branch Sync -- CRITICAL

### Rule
The AI rule set (`AGENTS.md`, `AI_GUIDELINES.md`, `.cursorrules`, `copilot-instructions.md`, `GAD.md`) represents the immutable "physical laws" of the repository ecosystem. Rules are global and MUST NOT vary between branches or projects.

### Authority Restriction
Only branches with the `ceo/` scope have the authority to modify rule files. Any rule change attempted on `infra/`, `feat/`, `fix/`, or other branches MUST be rejected by the AI Agent. Non-CEO branches MUST merge rule updates from a `ceo/` branch or from the `kuro-rules` master copy to maintain parity.

### Mandatory Sync Process
1. **Rule Modification**: When any rule is added or modified on a `ceo/` branch, the AI Agent MUST immediately sync the master `kuro-rules` repository and then propagate the rule to all active project copies.
2. **Review Enforcement**: No pull request can be merged without explicitly confirming that the branch carries the current rule set.
3. **Cross-Project Consistency**: Shared rule files MUST not drift across projects listed in `projects.txt`.

### Enforcement
```
IF a branch or project is on an outdated rule set:
  ACTION: STOP further rule work
  ACTION: Sync the current rule set from `kuro-rules`
  DO NOT: Continue with divergent rule files
```

---

## RULE 34: Strict Project Isolation -- MANDATORY

### Rule
When interacting with external tools such as Linear, GitHub, or search systems, the AI Agent MUST strictly limit its scope to the current active project context.

### Requirements
1. **Tool Filtering**: Always filter issues, projects, documents, and automations by the specific project name or project ID currently in scope.
2. **Context Integrity**: Do NOT read, comment on, or update unrelated projects unless they are explicitly cross-referenced for the current task.
3. **Choice Prompt**: If multiple projects are detected, the AI Agent MUST ask which project is active before continuing.

### Enforcement
```
IF search results or Linear issues contain multiple projects:
  ACTION: Filter to the active project only
  ACTION: Ask for clarification if the active project is ambiguous
  DO NOT: Mix work from multiple projects in the same response or change set
```

---

## RULE 35: CEO Progress Visibility in Linear -- MANDATORY

### Rule
The CEO's tasks, progress, and milestones MUST be automatically tracked and visible in Linear for all team members.

### Requirements
1. **Automatic Progress Updates**: Every action taken by the CEO on the `ceo/` branch MUST create or update a Linear issue.
2. **Real-time Visibility**: Team members can see:
   - What the CEO has completed (done)
   - What the CEO is working on (in progress)
   - What the CEO plans to do next (backlog)
3. **Milestone Tracking**: Each milestone (10%, 25%, 50%, 75%, 90%, 95%) MUST have a corresponding Linear issue.
4. **Workflow Integration**: The GitHub Actions workflow MUST update Linear automatically when rules are synced.

### Implementation
The CEO Progress workflow (`rule-sync.yml`) automatically:
- Creates/updates Linear issues for each milestone
- Posts comments on rule sync completion
- Tracks progress percentage in Linear
- Assigns label `CEO Decision` to all CEO items

### Verification Checklist
```
AT each session start:
  CHECK: Can you see CEO's current tasks in Linear?
  CHECK: Is progress percentage updated?
  CHECK: Are new milestones visible?
  IF NOT:
    ACTION: Trigger manual sync via workflow_dispatch
    ACTION: Report issue to DevOps
```

### Enforcement
```
IF CEO progress is NOT visible in Linear:
  ACTION: STOP - team cannot work without visibility
  ACTION: Verify Linear API keys are configured
  ACTION: Run workflow manually to sync
  DO NOT: Continue until visibility is restored
```

---

---

## RULE 36: Automated Session Status Report -- MANDATORY

### Rule
At the START of every AI session, the agent MUST report the complete status including:
1. Current branch
2. Last session progress
3. Pending tasks in Linear
4. Any blockers

### Requirements
1. **Session Gate**: Report status before any code work.
2. **Linear Integration**: Query Linear for CEO's current tasks.
3. **Progress Calculation**: Calculate pessimistic progress per Rule 3.
4. **Format**: Must include both French and English sections.

### Required Report Format
```markdown
## Francais
**Branche actuelle**: ceo/kuro-semantic-event-structures
**Taches CEO en cours**: [list from Linear]
**Taches terminees**: [list from Linear]
**Progression**: X% (pessimiste)
**Blockers**: [if any]

## English
**Current branch**: [branch name]
**CEO tasks in progress**: [list from Linear]
**CEO tasks completed**: [list from Linear]
**Progress**: X% (pessimistic)
**Blockers**: [if any]
```

### Enforcement
```
IF status report is NOT provided:
  ACTION: STOP - provide status report first
  DO NOT: Start any coding tasks
```

---

## RULE 37: CEO Real-Time Activity Sync to Linear -- CRITICAL

### Rule
The CEO's ALL activities, progress updates, and planned work MUST be automatically synchronized to Linear in real-time. Team members must always see:
- What the CEO completed (done)
- What the CEO is working on (in progress)
- What the CEO plans to do next (backlog)
- Current progress percentage
- Any blockers or impediments

### Requirements
1. **Automatic Issue Creation**: Every task, subtask, or work item MUST create a Linear issue automatically.
2. **Status Updates**: Status changes (todo -> in progress -> done) MUST be reflected in Linear immediately.
3. **Progress Tracking**: Progress percentage MUST be calculated and updated in Linear at each session end.
4. **Session Summary Sync**: Every SESSION_SUMMARY.md update MUST automatically update corresponding Linear issues.
5. **Milestone Progress**: Each milestone (10%, 25%, 50%, 75%, 90%, 95%) MUST have its own Linear issue with automatic progress updates.

### Implementation via GitHub Actions
The workflow `rule-sync.yml` MUST execute after EVERY session to sync:
- New tasks created during the session
- Tasks completed during the session
- Current progress percentage
- Session summary to Linear comments

### Required Linear Issue Types
| Issue Type | Description | Labels |
|------------|-------------|--------|
| Milestone | 10%, 25%, 50%, 75%, 90%, 95% tracking | CEO Decision, Milestone Task |
| Session Report | Each session's work and next steps | CEO Decision, Documentation |
| Blocker | Any impediment or roadblock | CEO Decision, Needs Review |
| Feature Task | Individual feature or rule work | CEO Decision, (appropriate label) |

### Verification Checklist
```
AT each session end:
  CHECK: Were all tasks created as Linear issues?
  CHECK: Did workflow sync progress to Linear?
  CHECK: Can team members see CEO's current status?
  CHECK: Is progress percentage accurate in Linear?
  IF NOT:
    ACTION: Manually sync immediately
    ACTION: Verify workflow configuration
```

### Enforcement
```
IF CEO activity is NOT visible in Linear:
  ACTION: STOP - team cannot work without visibility
  ACTION: Trigger manual sync via workflow_dispatch
  ACTION: Fix automation to prevent future gaps
  DO NOT: Continue until visibility is restored

IF progress is outdated (>24h old):
  ACTION: Update Linear immediately
  ACTION: Document reason for delay
  DO NOT: Allow stale progress data
```

---

## RULE 38: Mandatory Code Review After Commit -- MANDATORY

### Rule
A code review MUST be completed at EVERY push AFTER the commit but BEFORE any continuation of work on the rules. If no review is done, the continuation on the rules MUST be blocked.

### Requirements
1. **Review Tools**: Use one of the following automated or manual review tools:
   - **Qode** - AI-powered code review
   - **CodeRabbit** - Automated AI code review
   - **GitHub Pull Request Reviews** - Manual human review
   - **CodeClimate** - Automated quality analysis
   - Any equivalent code review tool

2. **Review Timing**:
   - Review MUST be completed AFTER the commit/push
   - Review MUST be completed BEFORE continuing work on the rules
   - Review MUST pass before proceeding with further rule modifications

3. **Review Criteria**:
   - Code quality and best practices
   - Security vulnerabilities (Rule 6)
   - Test coverage impact (Rule 5)
   - Regression prevention (Rule 18)
   - Documentation completeness

4. **Blocker Enforcement**:
   - If review is pending: STOP all new work on rules
   - If review fails: FIX issues before continuation
   - If no review done: BLOCK until review completed

### Verification Checklist
```
BEFORE continuing on rules:
  CHECK: Was a code review requested/completed after commit?
  CHECK: Did the review pass all checks?
  CHECK: Are there any pending issues to fix?
  IF review NOT done:
    ACTION: STOP immediately
    ACTION: Request/provide code review
    DO NOT: Continue working on rules
```

### Enforcement
```
IF code review NOT completed after commit:
  ACTION: STOP all rule development work
  ACTION: Request code review via Qode/CodeRabbit/GitHub
  ACTION: Wait for review approval
  DO NOT: Continue modifying rules
  DO NOT: Make new commits

IF review fails:
  ACTION: FIX identified issues
  ACTION: Request re-review
  DO NOT: Ignore review feedback
  DO NOT: Continue without fixing issues
```

---

## RULE 39: Pre-Marketing Pain-Point Due Diligence -- MANDATORY

### Rule
Before any marketing, outreach, landing page, waitlist, paid acquisition, or new product claim for a fresh hypothesis, the agent MUST run structured desk research to verify that the pain point is real, current, and safe to build around.

### Verification Checklist
```
BEFORE marketing or fresh build claims:
  1. RUN: The project-local Perplexity prompt in `prompts/perplexity.md`
  2. RUN: The project-local Grok prompt in `prompts/grok.md` or an equivalent X/blog/forum search
  3. COLLECT: At least 5 recent signals, with priority on 2025-2026 sources
  4. INCLUDE: At least 1 official or regulator source, 1 competitor or substitute signal, 1 recent blog/article, and 1 recent social or forum signal
  5. DOCUMENT: Problem existence, urgency, safety or compliance risk, platform dependency, and what does NOT prove willingness to pay
  6. VERDICT: State whether desk research is sufficient or whether 3-5 expert calls are still mandatory
```

### Enforcement
```
IF pain-point due diligence is missing:
  ACTION: STOP marketing and stop making factual claims about the problem
  ACTION: Run the research templates and document the result first
  DO NOT: Treat a pain point as validated from intuition alone

IF the remaining uncertainty is buying behavior, compliance, or integration:
  ACTION: Require 3-5 expert calls before stronger go-to-market claims
  DO NOT: Let desk research replace expert validation for those gaps
```

---

## RULE 40: CEO Complete Linear Dashboard Visibility -- MANDATORY

### Rule
The CEO MUST have complete, real-time visibility into ALL activities through Linear. Every action, progress update, and planned work MUST be automatically visible in Linear for the entire team.

### Requirements
1. **Complete Task Visibility**: All CEO tasks must be visible in Linear including:
   - Tasks completed (done)
   - Tasks in progress (in progress)
   - Tasks planned (backlog)
   - Blockers and impediments

2. **Automatic Real-Time Sync**: Every CEO action MUST automatically update Linear:
   - Task creation -> Linear issue created
   - Task start -> Linear status changed to "in progress"
   - Task completion -> Linear status changed to "done"
   - Session end -> Progress percentage updated in Linear

3. **Dashboard Integration**: The team MUST be able to see:
   - Current session status (what the CEO is working on now)
   - Next planned tasks
   - Progress percentage with breakdown
   - Any blockers

4. **Zero Manual Updates**: The CEO should NEVER manually update Linear. All updates MUST be automatic through:
   - GitHub Actions workflow (rule-sync.yml)
   - Automated scripts
   - MCP server integration

### Implementation
The workflow `rule-sync.yml` MUST include a CEO Activity Sync step:
- Query current branch status
- Calculate progress percentage
- List completed tasks from last session
- List in-progress tasks
- Identify blockers
- Update corresponding Linear issues automatically

### Verification Checklist
```
AT each session start:
  CHECK: Can you see ALL CEO tasks in Linear?
  CHECK: Is every task status accurate (done/in progress/backlog)?
  CHECK: Is progress percentage current?
  CHECK: Are blockers visible?
  IF ANY missing:
    ACTION: Fix sync automation immediately
    DO NOT: Continue until visibility is complete
```

### Enforcement
```
IF CEO cannot see complete task list in Linear:
  ACTION: STOP all work
  ACTION: Fix sync automation
  ACTION: Verify Linear API keys
  DO NOT: Proceed until visibility is complete

IF progress is outdated (>1 hour):
  ACTION: Trigger immediate sync
  ACTION: Verify workflow execution
  DO NOT: Allow stale data
```

---



## RULE 46: Regression and Mojibake Protection — MANDATORY

### Rule
AI agents MUST prevent regressions and encoding errors (Mojibake) by verifying the entire system state after any modification. Any use of special characters in French or other languages MUST be verified to maintain UTF-8 integrity.

### Enforcement
- REJECT: Any commit containing Mojibake or broken file encodings.
- ACTION: Use Python scripts for encoding repair if PowerShell or terminal-based redirection fails.

---

## RULE 47: History Preservation — MANDATORY

### Rule
Historical logs, specifically `SESSION_SUMMARY.md` and `SYNC_LOG.md`, MUST NEVER be overwritten or truncated. New entries MUST be prepended to the top of the file to maintain a continuous, chronological record of the project heritage.

### Verification
```
BEFORE updating SESSION_SUMMARY.md:
  1. READ existing content
  2. PREPARE new entry
  3. COMBINE by prepending new entry to existing content
  4. VERIFY: Previous summaries are still present at the bottom
  DO NOT: Use Overwrite=true/EmptyFile=true without preserving content
```

### Enforcement
```
IF history is deleted or overwritten:
  ACTION: STOP all work
  ACTION: RESTORE history from git or backups immediately
  ACTION: REPORT violation to user
  DO NOT: Continue as if the history loss is acceptable
```

