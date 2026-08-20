# Contributing to NeuralDBG

Thank you for considering contributing to NeuralDBG!

## Quick Start

```bash
make bootstrap
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
pytest tests/unit -q
```

## Development Workflow

1. Fork the repository and create a feature branch (`feat/my-feature` or `fix/my-fix`).
2. Make your changes with tests.
3. Ensure CI passes locally:
   ```bash
   pytest tests/unit --cov=neuraldbg --cov-fail-under=75 -q
   python -m bandit -r neuraldbg -ll
   ```
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Submit a pull request using the template in `.github/PR_TEMPLATES/default.md`.

## Branch Naming

See `RULE 30` in `DEV_RULES.md`: `feat/`, `fix/`, `docs/`, `chore/` prefixes.

## Code Style

- Python 3.9+ , typed (`mypy` optional)
- `ruff` for linting, `black` compatible formatting
- No `assert` in production paths (use explicit `raise`)
- Windows/Linux compatible (no hardcoded `/tmp`, handle `cp1252`)

## Testing

- Unit tests: `tests/unit/` (required, coverage gate 75%)
- Integration: `tests/` + root `validate_*.py` scripts
- 4-stage validation pipeline (R108): `stress_test_suite.py` → `validate_combinatorial.py` → `validate_oos.py` → `benchmark_honest.py`

Run the full pipeline:

```bash
python stress_test_suite.py
python validate_combinatorial.py --full
python validate_oos.py
python benchmark_honest.py
```

## Reporting Issues

Use the templates in `.github/ISSUE_TEMPLATE/`:

- `bug_report.md` — training failure not detected or false positive
- `false_positive.md` — spurious detection on healthy run
- `feature_request.md` — new architecture or failure type

For security issues, see `SECURITY.md` — do not file public issues.

## PyTorch Compatibility

Tested against PyTorch 2.0.1 → 2.11.0+ (see `.github/workflows/ci.yml` matrix). When adding hooks, verify:

- `register_full_backward_hook` vs `register_backward_hook` semantics
- `torch.compile` compatibility (operate at module boundaries)
- RNN tuple outputs `(output, (h_n, c_n))`

## Release Process

Releases follow SemVer (`CHANGELOG.md`). Cadence: monthly minor, patch as needed for fixes. See `GOVERNANCE.md` for decision process.

## Questions?

Open a discussion or contact `neuraldbg@lemniscate.ai`.
