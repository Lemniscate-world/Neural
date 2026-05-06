# RULE 83: Investor-Ready Discord Summary

Trigger: at each session producing status/progress/decision change.

---

## Purpose

Generate a compact, investor-ready summary at the end of every significant session.
This summary must be publishable as-is on Discord or any investor-facing channel.

---

## Format

- **Length**: 3-5 sentences maximum.
- **Tone**: Direct, minimal jargon, no filler.
- **Language**: French or English depending on target audience.
- **Content**:
  1. What changed this session (status, progress %, decision).
  2. Key numbers (metrics, responses, coverage, revenue).
  3. Immediate next step with a deadline if possible.

---

## Example

```
Hermes validation: landing page deployed, endpoint configured.
Progress 8% -> 10%. Blocker cleared: form endpoint live.
Next: collect 20+ merchant responses within 7 days.
Decision gate: GO/ADJUST/NO-GO due May 8.
```

---

## Enforcement

- MUST be generated before SESSION_SUMMARY.md is considered complete.
- MUST be included in SESSION_SUMMARY.md under a dedicated section.
- MUST be copied to `kuro-rules/MARKETING_MEMORY/` if the project has one.
- MUST NOT contain emojis (R9).
- MUST NOT contain aspirational claims without numbers.

---

## Failure Mode

If no investor summary is produced:
- The session is incomplete.
- No new feature work should start until it is written.
