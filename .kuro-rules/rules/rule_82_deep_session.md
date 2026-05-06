# RULE 82: Deep Session Summary — Full Detail

## Purpose
Eliminate the "reprise tax" — the cognitive cost of resuming a project.
A session summary must allow a fresh agent to resume in < 2 minutes, zero questions asked.
Think of it as a pilot pre-flight checklist left for the next pilot.

## When to use Deep Format
- Any session where code was written or architecture was decided
- Any session > 1 hour
- Any session ending mid-task (WIP state)
- ANY session on OpenQuant, NeuroDose, NeuralDbg, or other active technical projects

## The Deep Session Summary Template

---
# Session Summary - YYYY-MM-DD
**Editor**: [IDE]
**Branch**: [branch name]
**Duration**: [X hours]

## RESUME IN 30 SECONDS
[3 sentences max. What is the project, where is it right now, and what is the single next action.]
Example: "OpenQuant is a quant trading bot for Brent-WTI. Today we validated Kalman 2D with Sharpe 5.90 on Oct 2024 window. Next: implement volatility targeting in kalman_pair.py > _calculate_position_size()."

## EXACT CURSOR POSITION
**File**: [path/to/file.py]
**Function/Class**: [function_name or ClassName.method_name]
**What was in progress**: [one sentence — e.g., "Writing the volatility targeting logic, stopped after the entry condition, before the exit"]
**Line context**: [paste the last 3-5 lines written, or the TODO/FIXME left in code]

## FIRST COMMAND TO RUN
```
[exact command to run first — e.g., "cd ~/Documents/OpenQuant && python -m pytest tests/test_kalman.py -v"]
```
**Expected output**: [what success looks like]
**If it fails**: [what to check first]

## DECISION LOG (why we chose X over Y)
| Decision | Option Chosen | Rejected Options | Reason |
|----------|--------------|------------------|--------|
| [e.g., Kalman model] | 2D State-Space | 1D Static | 1D cannot handle structural drift in Brent-WTI spread |
| [e.g., Data source] | yfinance | Bloomberg API | Bloomberg requires paid subscription |

## ARCHITECTURE SNAPSHOT (current state)
```
project/
  src/
    [file.py] - [what it does NOW, after today's changes]
    [file.py] - [status: STABLE / WIP / BROKEN-INTENTIONAL]
  tests/
    [test_file.py] - [X passing / Y failing / reason for failures]
```
Focus on what CHANGED this session. Don't repeat unchanged structure.

## DANGER ZONES (do not touch without reading)
| File/Function | Why dangerous | What to do instead |
|---------------|---------------|--------------------|
| [kalman_pair.py > _fit()] | Half-refactored, breaks if called directly | Use _fit_safe() wrapper |
| [config.py line 47] | Hardcoded window size, will be parameterized next session | Leave as-is |

## OPEN QUESTIONS (unresolved decisions)
1. [Question] — [Context] — [Options being considered]
   Example: "Should we use Kalman innovation variance or rolling std for dynamic threshold? Kalman is theoretically cleaner but harder to debug."

## ENVIRONMENT STATE
**Branch**: [branch name] (on remote? YES/NO)
**Python/venv**: [active? path?]
**Missing dependencies**: [e.g., "bandit not installed — `pip install bandit`"]
**Known env issues**: [e.g., "MT5 connection requires VPN"]
**Last test run result**: X passing, Y failing — [brief reason for failures]

## PROGRESS & VALIDATION
**Progress**: X% (R3 pessimistic formula)
**Cry Test**: X% (R71)
**Validation gate**: [current gate — e.g., "L2: need 3 expert pharmacist calls"]
**VALIDATION_PASSED**: [YES - Milestone X% / NO - locked at X%]

## COMPACT SUMMARY (for human reading, 150 words max)

### Francais
[150 words max, plain language, no jargon]

### English
[150 words max, plain language, no jargon]

---

## Anti-patterns (what the old format did wrong)

### TOO VAGUE (forbidden)
- "Worked on the Kalman model"
- "Next steps: continue backtest"
- "Files changed: robot.py"

### CORRECT (required)
- "Modified `kalman_pair.py > _calculate_position_size()` to add 2D state-space beta. Stopped after entry condition. Exit condition not written yet."
- "Next: run `python research/optimize_2d_kalman.py --window 2024-10 --param-grid configs/grid_v2.json`"
- "Changed `kalman_pair.py` lines 142-187: replaced static beta with dynamic beta from Kalman state vector. Lines 188-210 are TODO."

## Enforcement
IF session ends without EXACT CURSOR POSITION: the summary is incomplete.
IF FIRST COMMAND TO RUN is missing: the summary is incomplete.
IF DECISION LOG is empty for a session with architectural choices: the summary is incomplete.

The test: give the summary to a fresh agent with zero context. Can it resume without asking you a single question? If no: rewrite until yes.

## Relation to other rules
- R4: this rule EXTENDS R4, does not replace it. Deep format adds the technical sections.
- R73: when resuming, agent reads EXACT CURSOR POSITION first, then FIRST COMMAND TO RUN.
- R79: Compact Summary section satisfies R79 (150 words max).
- R17: DECISION LOG partially satisfies R17 (documents reasoning for phase transitions).