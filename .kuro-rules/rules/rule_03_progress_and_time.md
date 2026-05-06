# RULE 3.5: AI Time Estimation - Full Detail

## Core Principle
Never give deadlines in weeks/months for first estimation. Always decompose into hours/days first.

## MVP Timeline Framework
| Phase | Duration | What it means |
|-------|----------|---------------|
| Prototype | 2-4 days | Core mechanic working, no polish |
| MVP | 2-4 weeks | Usable by 1 real user end-to-end |
| Beta | 4-8 weeks | Usable by 10+ users, feedback loop active |
| V1 | 2-3 months | Stable, documented, distributable |

## Estimation Rules
- Break every feature into tasks <= 4h each
- If a task estimate > 4h, split it
- Add 30% buffer to any estimate
- State estimates as ranges: "2-4 hours" not "3 hours"
- Never say "a few days" - say "2-3 days"

## Red Flags (recalculate if these appear)
- Any single task > 8h -> not decomposed enough
- MVP estimate > 4 weeks for solo dev -> scope too large
- No buffer included -> subtract 15%

## Enforcement
IF user asks "how long will X take?":
  1. Break X into sub-tasks
  2. Estimate each in hours
  3. Sum + 30% buffer
  4. Give range, not point estimate
  DO NOT: Give a single number with false precision
# RULE 3: Progress Tracking - Full Detail

## Formula
Progress = (Sum of Weight_i * Multiplier_i) - Total_Debt

## Baseline Weights (CEO/Product persona)
| Component | Weight | Complete when |
|-----------|--------|--------------|
| Mom Test | 10% | 5+ interviews, results + decision.md signed |
| Core: Data Extraction | 10% | Working, zero-overhead |
| Core: Semantic Logic | 10% | Raw data -> semantic events |
| Core: Reasoning Engine | 10% | Causal inference, hypothesis ranking |
| Core: Scale/Robustness | 10% | Stress tested, zero memory leaks |
| Test Coverage (60%+) | 20% | pytest --cov shows >= 60% |
| Security & Hardening | 10% | bandit 0 issues, safety 0 critical |
| CI/CD & DevOps | 10% | Pipeline passing on main |
| Documentation & Mkt | 10% | README, CHANGELOG, API docs, 1+ marketing metric |

## Infrastructure Grid (DevOps/MLOps persona)
| Component | Weight |
|-----------|--------|
| Assigned Linear tasks completed | 50% |
| Test coverage on infra code (60%+) | 20% |
| Security scans pass | 15% |
| Documentation updated | 15% |

## Quality Multipliers
- 1.0x: Fully documented, tested, optimized
- 0.8x: Functional and tested, lacking deep docs
- 0.5x: Functional but hacky, no tests

## Validation Debt (subtract from total)
- -10% persistent optimism bias
- -5% per milestone reached without VALIDATION_PASSED
- -5% if security scans missing or red
- -5% if R17 skipped for phase transition

## Sanity check
"If this feature were deleted today, what % of the MVP would be missing?"

## Enforcement
Never inflate progress to make user happy.
STOP and recalculate if % > actual completion.
