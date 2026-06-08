# NeuralSuite — Integration Contract

> MID: ECO-001
> Owner: LambdaSection
> Status: ACTIVE
> Last updated: 2026-06-07

## Vision

**NeuralSuite** is the unified brand for three complementary tools that diagnose and fix deep learning training failures.

```
┌─────────────────┐    events JSON    ┌─────────────────┐
│   NeuralDBG     │ ────────────────▶ │   Aquarium      │
│ (diagnostic)    │                   │ (visualizer)    │
└────────┬────────┘                   └─────────────────┘
         │ explain_failure()
         │ CausalHypothesis
         ▼
┌─────────────────┐
│  Neural-Agent   │  (auto-corrector)
└─────────────────┘
         │ remediation rules
         ▼
   training loop patched
```

## The Three Components

### 1. NeuralDBG — Diagnostic Engine
- **Package**: `neuraldbg` (PyPI)
- **Role**: Instrument PyTorch training, capture semantic events, produce causal hypotheses + JSON export
- **Output**: `dbg.explain_failure()` -> list[CausalHypothesis], `dbg.export_json()` -> events.json
- **Status**: v1.3.1 published, v1.3.2 in dev

### 2. Neural-Agent — Auto-Corrector
- **Package**: `neural-agent` (PyPI)
- **Role**: Consume NeuralDBG hypotheses, apply remediation rules, patch training scripts
- **Input**: `dbg.explain_failure()` (Python objects, in-process)
- **Output**: `remediation_applied` event + patched optimizer/model state
- **Status**: Pipeline built (87 tests), model not yet trained

### 3. Aquarium — Visualizer
- **Package**: Desktop app (Tauri)
- **Role**: Consume NeuralDBG JSON exports, render interactive causal trees
- **Input**: `events.json` (stable, versioned schema)
- **Status**: Export validated, MVP delivered, dormant

## Inter-Component Contracts

### NeuralDBG -> Neural-Agent (in-process, Python)
```python
from neuraldbg import NeuralDbg
from neuralagent import RemediationRunner

with NeuralDbg(model) as dbg:
    runner = RemediationRunner(dbg)
    runner.register_default_rules()
    for step in range(N):
        loss = train_step(x, y)
        loss.backward()
        dbg.record_loss(loss.item())
```

### NeuralDBG -> Aquarium (out-of-process, JSON)
```bash
python train.py --neuraldbg-export events.json
# Open Aquarium, drag events.json, view causal graph
```

### Dependency Direction
`neural-agent` depends on `neuraldbg`. Not the other way around. NeuralDBG MUST remain usable without Neural-Agent installed.

## Branding Rules

- Public-facing: always say "NeuralSuite" as the umbrella brand
- Package names stay: `neuraldbg`, `neural-agent`, `aquarium` (no rename)
- README titles: "NeuralSuite" with subtitle explaining the component
- Social posts: "#NeuralSuite" hashtag, mention components individually when needed
- PyPI descriptions: "Part of the NeuralSuite ecosystem"

## Roadmap Integration

| Milestone | NeuralDBG | Neural-Agent | Aquarium |
|-----------|-----------|--------------|----------|
| M1 (June) | v1.3.2 tag, benchmark | MHA rule wired | Reads BUG-001 JSON |
| M2 (July) | 5+ scenarios benchmark | Published PyPI, 10 rules | Dashboard bugs catalog |
| M3 (August) | 20+ bugs catalog | 25 rules, autonomous | Visual bug catalog |
