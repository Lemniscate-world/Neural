# COMPATIBILITY_MATRIX.md — NeuralSuite

> Cross-repo SemVer matrix for the NeuralSuite ecosystem.
> MANDATORY per R105. Update on every breaking change in a shared interface.

## Versions

| Repo                  | Current | Released     | Distribution                  | Owner class (R87) |
|-----------------------|--------:|--------------|-------------------------------|-------------------|
| `NeuralDBG`           |  1.3.2  | 2026-05-20 (1.3.1) | Public PyPI `neuraldbg`       | OWNED (LambdaSection) |
| `Neural-Agent`        |  0.1.0  | not yet (dev)     | Private (closed beta)         | OWNED (LambdaSection) |
| `Aquarium`            |  0.1.0  | MVP delivered     | Source (Tauri desktop)        | OWNED (LambdaSection) |
| `NeuralDBG-Engine`    |  1.0.0  | 2026-06 (pkg)     | GitHub Packages (private)     | OWNED (LambdaSection, private) |

## Pairwise Compatibility

| Consumer \ Provider       | neuraldbg ≥1.3.0 | neuraldbg-engine ≥1.0.0 | events.json schema v1 |
|---------------------------|:----------------:|:-----------------------:|:---------------------:|
| `neural-agent` ≥0.1.0     | ✅ compatible    | n/a (consumes dbg API)  | n/a (in-process)      |
| `aquarium` ≥0.1.0         | ✅ (reads JSON)  | n/a                     | ✅ strict (see schema/events.json) |
| `neuraldbg` w/ `neuraldbg-engine` | n/a        | ✅ compatible (1.0.0+)  | n/a                   |
| `neuraldbg` w/o engine    | n/a              | ✅ graceful fallback    | n/a                   |

**Status legend**: ✅ compatible — 🟡 breaking-pending — ❌ incompatible

## Shared Interface Contracts

### 1. `dbg.explain_failure() -> list[CausalHypothesis]` (Python, in-process)
- **Owner**: `NeuralDBG`
- **Consumers**: `Neural-Agent`
- **Stability**: stable since 1.3.0
- **Required methods on `CausalHypothesis`**: `failure_type: str`, `root_cause_layer: str | None`, `root_cause_step: int | None`, `confidence: float`, `description: str`, `evidence: list[str]`, `remediation_hint: str | None`
- **Bump rule**: any change to field types / removal = MAJOR bump in `neuraldbg` + MAJOR bump in `neural-agent`

### 2. `events.json` (JSON, out-of-process)
- **Owner**: `NeuralDBG` (writes), `Aquarium` (reads)
- **Schema file**: `neuraldbg/schema/events.json`
- **Stability**: versioned, see `schema_version` field
- **Bump rule**: any required field added = MINOR bump + Aquarium update; any field removed/renamed = MAJOR bump + Aquarium update

### 3. `NeuralDBG-Engine` import contract (Python, in-process, optional)
- **Owner**: `NeuralDBG-Engine` (writes), `NeuralDBG` (reads)
- **Discovery**: `importlib.util.find_spec("neuraldbg_engine")` + `from neuraldbg_engine import CausalEngine`
- **Required class**: `CausalEngine(dbg)` with methods:
  - `detect_gradient_transition(prev_norm, current_norm)`
  - `classify_gradient_health(norm)`
  - `classify_activation_health(stats)`
- **Bump rule**: any method signature change = MAJOR bump in `neuraldbg-engine` + MINOR bump in `neuraldbg` core

## Last Integration Test

| Date       | Test                                    | Result |
|------------|-----------------------------------------|--------|
| 2026-06-08 | `tests/integration/test_lstm_demo.py`   | ✅ pass |
| 2026-06-08 | `tests/integration/test_gan_demo.py`     | ✅ pass |
| 2026-06-08 | `tests/integration/test_torch_compile_demo.py` | ✅ pass |
| 2026-06-08 | `tests/integration/test_critical_scenarios.py` | ✅ pass |

## Required Upgrade Paths

### If `neuraldbg` moves to 2.0.0 (breaking):
- `neural-agent` must move to 0.2.0 (consume new API)
- `aquarium` must move to 0.2.0 (read new JSON schema)
- `neuraldbg-engine` must be re-validated against new core (compatibility patch if needed)

### If `events.json` schema moves to v2:
- `neuraldbg` must write v2
- `aquarium` must read v2 (with v1 → v2 migration helper)

## Sync Coordination

- **Branch strategy**: per R30 — each repo uses its own trunk (`main`), features in `feat/*`, fixes in `fix/*`
- **Tag coordination**: SemVer tags on each repo, no lock-step. Cross-repo compatibility validated by integration test suite in `NeuralDBG/tests/integration/`
- **CI cross-repo**: triggered manually for now (no monorepo CI). Run from `NeuralDBG/` after pulling latest `neural-agent` and `neuraldbg-engine` tags.
