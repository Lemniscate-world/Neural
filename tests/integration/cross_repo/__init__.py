"""Cross-repo integration tests (R105).

Validates that NeuralDBG works correctly with the other components of the
NeuralSuite ecosystem:

- **neural-agent** : consumes `dbg.explain_failure()` and applies remediations
- **neuraldbg-engine** : optional upgrade, loaded via importlib
- **aquarium** : reads `events.json` exports

These tests are NOT run in the default CI (they require sibling repos
cloned as siblings of this one). They are run manually via:

    pytest tests/integration/cross_repo/ -m cross_repo
    pytest tests/integration/cross_repo/ -m cross_repo --run-engine

Status: skeleton only — see TODOs in each test file.
"""
