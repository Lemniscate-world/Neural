# Project-local Grok prompt

Use this prompt for tweets, blogs, forums, founder commentary, and recent public chatter before marketing a new pain point.

```text
I need pre-marketing pain-point due diligence for a startup idea.

Project:
- Name: NeuralDBG
- Geo: TBD
- Target user: TBD
- Buyer hypothesis: TBD
- Pain point hypothesis: TBD
- Proposed product or wedge: TBD
- Main alternatives today: TBD

Validation stage:
- Current stage: L0

Mission:
- Search recent public signals from 2025-2026.
- Search in French and English.
- Prioritize X posts, LinkedIn posts, blog posts, local press, founder commentary, forum threads, Reddit, and app reviews if relevant.
- Look for:
  - user complaints
  - recurring pain language
  - process failures
  - integration pain
  - compliance or platform risk
  - competitor weakness
  - public signs of urgency

Rules:
- Every signal must include a direct URL and date.
- If a signal is weak or anecdotal, label it weak.
- Do not present social chatter as proof of willingness to pay.
- Separate operational pain from generic hype.

Return:

1. Signal log
- A table with:
  date | source_type | source_url | actor | country | signal | strength | what_it_suggests | why_it_is_not_enough

2. Pain-point synthesis
- Say whether public chatter points to a real operational pain, a niche complaint, or generic market noise.

3. Risk and safety synthesis
- Flag platform dependency, compliance risk, or reputational risk that would make this pain point unsafe to market aggressively.

4. Competitor and substitute synthesis
- Identify whether the real substitute is an incumbent vendor, internal team, spreadsheet plus email, agency, integrator, or "do nothing".

5. Validation-stage verdict
- What can public-signal research confirm at the current stage?
- What can it NOT confirm?
- Which expert calls are still mandatory before the next stage?
```

