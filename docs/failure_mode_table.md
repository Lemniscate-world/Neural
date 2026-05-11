# Failure Mode Table - NeuralDBG

| Risk | Probability | Impact | Evidence FOR | Evidence AGAINST | Remedy | Kill Condition |
|------|-------------|--------|--------------|------------------|--------|----------------|
| **Structural Fragility** | High | Critical | Core engine file (`neuraldbg.py`) was lost in a merge without detection. | Post-merge cleanup and modular refactor completed. | Implement Rule 84 and pre-commit hard-locks. | 2nd occurrence of core data loss. |
| **Governance Fragility** | Medium | High | SemVer rules (Rule 19) were ignored (missing `-kuro` suffix). | Rules are documented and indexed in `AGENTS.md`. | Automation of release tagging via `Makefile`. | Repeated failure to adhere to SemVer-Author. |
| **Security Debt** | Low | High | Bandit audit flagged `assert` in core engine. | High test coverage (103 tests) and psutil is optional. | Replace `assert` with proper exceptions in production code. | Security breach in automated pipeline. |
| **Dependency Vulnerabilities** | Medium | High | 11 vulnerabilities detected by `safety` in upstream dependencies (mlflow, etc.). | Monitoring enabled via `pre-commit`. | Migration plan to patched versions. | Exploitable RCE in production. |
| **Versioning Inconsistency** | High | Medium | `pyproject.toml` was at `0.1.0` while `CHANGELOG` was at `1.2.0`. | Versioning is now synced to `1.2.0-kuro`. | Centralized versioning in `pyproject.toml`. | Version drift across 2+ files. |
| **Technical Debt (Dependency)** | Medium | Medium | 4+ requirements files with overlapping dependencies. | DVC is initialized for binary tracking. | Consolidate all requirements into `pyproject.toml`. | Dependency hell blocking `make bootstrap`. |
