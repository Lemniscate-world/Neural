# NeuralSuite — Integration Contract

> MID: ECO-001
> Owner: LambdaSection
> Status: ACTIVE
> Last updated: 2026-06-09

## Vision

**NeuralSuite** is the unified brand for the toolkit that diagnoses and fixes deep learning training failures.

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

         ▲ optional upgrade
┌─────────────────┐
│ neuraldbg-engine│  (advanced causal inference, proprietary)
└─────────────────┘
```

## The Components

### 1. NeuralDBG — Diagnostic Engine
- **Package**: `neuraldbg` (PyPI)
- **Role**: Instrument PyTorch training, capture semantic events, produce causal hypotheses + JSON export
- **Output**: `dbg.explain_failure()` -> list[CausalHypothesis], `dbg.export_json()` -> events.json
- **Status**: v1.3.1 published, v1.3.2 in dev
- **Works without**: `neural-agent`, `neuraldbg-engine` (core fallbacks cover common cases)

### 2. Neural-Agent — Auto-Corrector
- **Package**: `neural-agent` (PyPI)
- **Role**: Consume NeuralDBG hypotheses, apply remediation rules, patch training scripts
- **Input**: `dbg.explain_failure()` (Python objects, in-process)
- **Output**: `remediation_applied` event + patched optimizer/model state
- **Status**: Pipeline built (87 tests), model not yet trained
- **Distribution**: Closed beta (not on public PyPI yet)

### 3. Aquarium — Visualizer
- **Package**: Desktop app (Tauri)
- **Role**: Consume NeuralDBG JSON exports, render interactive causal trees
- **Input**: `events.json` (stable, versioned schema)
- **Status**: Export validated, MVP delivered, dormant

### 4. neuraldbg-engine — Advanced Causal Inference (optional)
- **Package**: `neuraldbg-engine` (GitHub Packages, private registry)
- **Role**: Pluggable upgrade for NeuralDBG. Adds advanced heuristics: data anomaly detection, optimizer instability, cross-architecture coupling logic. Powers the closed-beta diagnostics.
- **Interface**: Same `dbg.explain_failure()` API — no user code change required
- **Status**: v1.0.0 packaged, distributed via private registry
- **Discovery**: NeuralDBG uses it opportunistically (`importlib` conditional import). Core fallbacks cover the open-source path.

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

### NeuralDBG <-> neuraldbg-engine (optional, in-process)
```python
# NeuralDBG auto-detects neuraldbg-engine if installed.
# When present: richer hypotheses (coupling, transitions, optimizer instability).
# When absent:  fallbacks return [] (no crash) — see cdp_protocol_definition.md.
```

### Dependency Direction
`neural-agent` depends on `neuraldbg`. `neuraldbg-engine` is loaded optionally by `neuraldbg`. Neither is required for the others to work.

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
