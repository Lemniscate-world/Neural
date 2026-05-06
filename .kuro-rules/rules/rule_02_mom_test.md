# RULE 2: Mom Test Gate - Full Detail

## Project Classification (ask FIRST)
- client-requested -> R2 = N/A, proceed
- verified-problem -> R2 = N/A, proceed
- personal -> R2 = N/A, proceed
- startup -> R2 ACTIVE, read below

## Gate: 0-10% phase
No production code. Allowed: interviews, mom_test_script.md, mom_test_results.md, decision.md, brainstorm in ideas.md.

## Default: 5 Mom Test interviews
Files required: mom_test_script.md, mom_test_results.md, decision.md
Criteria: 3+ spontaneous pain mentions, 2+ solution seekers.

## Alternatives (with user approval + documented in validation_evidence.md)
| Method | Required Evidence | Equivalent |
|--------|-------------------|-----------|
| Working MVP | 2+ active users | 2 interviews |
| Landing page + waitlist | 100+ signups, 3-5% conversion | 3 interviews |
| Wizard of Oz | 5+ simulated sessions | 3 interviews |
| Pre-orders | 100% funding goal | 4 interviews |
| Observed behavior | 10+ documented cases | 2 interviews |
| Desk Research (R75) | 10+ verified sources, 3+ behavioral proofs | 2 interviews |
| Automated L2 validation | See below | 2-3 interviews |

## Automated L2 Methods

### Job Market Scraping
Scrape: Welcome to the Jungle, LinkedIn Jobs, Indeed
Thresholds: 20+ postings/week (L2), 50+ (L3), 60% with salary (L2)
Equivalence: 50+ postings with salary = 2 interviews

### Landing Page Auto-Qualification
Typeform/Google Forms: role, company size, pain ranking, budget, beta interest
Thresholds: 10+ qualified responses (L2), 20+ (L3)
Equivalence: 20+ qualified = 3 interviews

### Community Listening Bot (experimental)
Monitor: r/devops, Discord K8s, Discord DevOps FR
Thresholds: 5+ pain mentions/week, 3+ conversations, 2+ explicit solution-seeking
Equivalence: 10+ natural pain mentions = 2 interviews

## Hybrid Recommended
L1: Desk Research (10+ sources) = 2 interviews
L2: Job scraping + landing page = 2-3 interviews
Total: 4-5 equivalent = validation complete

## Enforcement
IF startup AND checklist incomplete: STOP, SET progress <= 10%, report missing items.
IF client/verified/personal: mark N/A, proceed without Mom Test artifacts.

## Desk Research as Alternative
Requires: 10+ verifiable sources, 3+ behavioral proofs, 2025-2026 dates, documented in validation_evidence.md.
Behavioral proofs: A=pain quote, B=workaround built, C=willingness to pay, D=business impact

## Pivot Process (1-day)
1. Desk Research 4h: 10+ sources, GO/NO-GO
2. MVP Prototype 4h: 1 feature, working demo
3. Decision: GO / NO-GO / ADJUST