# DEC-001 — Stop HN retry; use channel basket as L2 proxy

> MID: DEC-001
> Date: 2026-05-29
> Source: docs/launch_postmortem.md
> Status: ACCEPTED

## Context

HN launch of NeuralDBG v1.3.0 (2026-05-18) was blocked at submission. Account-level anti-abuse filter, no public exposure. 0 upvotes, 0 comments, 0 emails captured. External, not actionable.

## Decision

Stop treating HN as the L2 validation gate. Use a **channel basket** as proxy:
- PyTorch Forums (discuss.pytorch.org)
- Reddit (r/PyTorch, r/MachineLearning, r/deeplearning)
- GitHub issue templates + Discussions
- ML Discords (FrancophonIA, PyTorch)
- Lobsters (already active)

## Rationale

- Karma/account-age filters on HN are **external to the product**. No engineering response fixes them.
- Strongest signal is **qualified technical feedback** that becomes an issue, PR, or reproducible failing training loop — not upvotes.
- A single blocked channel does not invalidate the launch hypothesis (R98).

## Consequences

- Re-attempt HN only when (a) account has visible history, or (b) a trusted proxy submits it.
- Add a non-marketing GH CTA: "Open an issue with a failing training loop and expected diagnosis".
- Track signal in `docs/tracking/acquisition_tracker.md`.

## References

- `docs/launch_postmortem.md`
- R97 (Launch Planning), R98 (Pre-Launch MVP Verification), R99 (Acquisition Tracker)
