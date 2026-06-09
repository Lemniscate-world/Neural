# Cross-repo integration tests (R105)

This directory holds cross-repo tests for the NeuralSuite ecosystem.
They are **not** part of the default CI because they require sibling repos.

## Why this exists

Per R105, when NeuralDBG ships a change to a shared interface (`dbg.explain_failure()`,
`events.json` schema, the `neuraldbg-engine` discovery contract), we MUST verify
that consumers (`neural-agent`, `neuraldbg-engine`, `aquarium`) still work.

Running these tests locally:
```bash
# 1. Clone sibling repos (one level up from NeuralDBG/)
#    ~/Documents/NeuralDBG/
#    ~/Documents/Neural-Agent/
#    ~/Documents/NeuralDBG-Engine/
#    ~/Documents/Aquarium/

# 2. Install each in editable mode
cd ../Neural-Agent && pip install -e ".[dev]" && cd -
cd ../NeuralDBG-Engine && pip install -e ".[dev]" && cd -

# 3. Run cross-repo tests
pytest tests/integration/cross_repo/ -m cross_repo -v
```

## Status: SKELETON

| Test file | Status | Effort |
|-----------|--------|--------|
| `test_neuraldbg_neuralagent.py` | Skeleton + TODOs | ~1-2h to flesh out |
| `test_neuraldbg_engine.py` | Skeleton + TODOs | ~1-2h to flesh out |
| `test_neuraldbg_aquarium.py` | Not started | ~1h (just JSON schema) |

## Markers

All tests in this directory are marked with `@pytest.mark.cross_repo`.
They are excluded from the default `make test` target.
