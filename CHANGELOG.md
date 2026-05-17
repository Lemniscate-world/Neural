# Changelog

All notable changes to NeuralDBG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 2 dogfooding: LSTM/Time Series failure scenarios (vanishing recurrent, exploding recurrent, deep LSTM)
- Phase 2 dogfooding: GNN (GCN/GAT) failure scenarios (oversmoothing, exploding, NaN injection)
- Phase 2 dogfooding: torch.compile (Dynamo) compatibility scenarios (healthy, vanishing, exploding)
- Phase 2 dogfooding: RL (PPO-style) failure scenarios (policy collapse, value explosion, reward hacking)
- Phase 2 dogfooding: Distributed/DataParallel failure scenarios (healthy, vanishing, exploding under DP)
- Engine fallbacks in core: `_classify_activation_health`, `explain_failure`, `detect_coupled_failures`, `export_mermaid_causal_graph`, `_classify_data_health`, `_check_data_anomaly` now work without proprietary engine
- **Phase 3**: Complete Aquarium JSON export schema with all required fields (events, hypotheses, couplings, first_failure_layer, first_failure_step, loss_history)
- 14 new unit tests for Aquarium export (`test_aquarium_export.py`)
- Aquarium export integrated into LSTM demo with auto-export to `aquarium_exports/`

### Phase 7 — Two-Package Architecture ✅
- Repo `neuraldbg-engine/` confirmé existant (`~/Documents/NeuralDBG-Engine`) avec structure complète
- Import conditionnel fonctionnel : `neuraldbg` (public) + `neuraldbg-engine` (privé)
- **130 tests passent avec engine installé**, fallbacks fonctionnent sans engine
- Architecture validée : core MIT open-source, engine propriétaire séparé

### Phase 5 — Publication PyPI ✅
- `pyproject.toml` mis à jour : auteurs, keywords, classifiers, license SPDX
- GitHub Actions workflow `publish.yml` : build → TestPyPI (manual) → PyPI (release)
- Package buildé : `neuraldbg-1.3.0.tar.gz` + `neuraldbg-1.3.0-py3-none-any.whl`
- Twine check : PASSED
- Test install fresh venv : import réussi

## [1.3.0] - 2026-05-14
### Added
- ResNet-18 failure scenarios demo (`demo_resnet_failures.py`) : vanishing gradients (Tanh + small init), exploding gradients (high LR), data anomaly (NaN injection)
- Integration tests for ResNet-18 demo (5 tests, 100% coverage)
- Semantic demo smoke tests (`test_semantic_demo.py`) for causal hypothesis validation

### Fixed
- Deduplication of logical causal couplings in `detect_coupled_failures()` and Mermaid graph export
- Import path for MLflow demo test after directory restructuring
- Graceful degradation of CPU resource sampler after psutil failure (avoids repeated exceptions)

## [1.2.0] - 2026-05-11
### Added
- Integrated PR 651: Detect Python version mismatch in `ensure_venv.sh` (MLO-17).
- Integrated PR 652: Initialized DVC for binary artifact versioning (MLO-4).
- Integrated PR 654: Resource profiling (CPU/GPU memory) integration for semantic events (MLO-10).
- Integrated PR 656: `SESSION_SUMMARY.md` to `.docx` conversion tool (NDBG-5).

### Changed
- Refactored repository structure: Unified all scripts into `infrastructure/scripts/`.
- Moved `neuraldbg.py` to `neuraldbg/__init__.py` for better package organization.
- Standardized `Makefile` to use centralized infrastructure scripts.
- Cleaned root directory by moving legacy security reports to `outputs/reports/`.

### Fixed
- Restored `neuraldbg.py` core engine which was incorrectly removed in previous refactor commits.
- Fixed import paths in test suite after directory restructuring.
- Resolved multiple merge conflicts in `.gitignore` and `Makefile`.

### Added
- `scripts/publish_session_summary_to_gdocs.py` to publish `SESSION_SUMMARY.md` directly to Google Docs (append/replace modes)
- `.github/workflows/publish-summary-to-google-docs.yml` to support scheduled/manual Google Docs sync from CI with secrets-based auth
- `GOOGLE_DOCS_SYNC.md` setup guide for Google Workspace service account integration
- Rule 39 (`CI/CD Debugging First`) synchronized across AI rule files
- Mandatory product & quality rules in `.cursorrules`, `ia_rules/AI_GUIDELINES.md`, `.github/copilot-instructions.md`, and `.cursor/rules/product-quality.mdc`
- Strategic section "Tools for the AI Era" explaining why structured tools matter when AI agents can code
- `.github/workflows/codeql.yml` — CodeQL security analysis (Python)
- `.github/workflows/codacy.yml` — Codacy static analysis (auto-detects Python)
- `.antigravity/RULES.md` — Copie des règles pour l’IDE AntiGravity uniquement
- `PROJECTS.md` — Roadmap Projets A & B (racine, aucun lien avec AntiGravity)
- `artifacts/` — Artifacts générés (déplacés depuis .antigravity/artifacts)

### Changed
- Projet A : repo dédié sous Quant-Search, NeuralDBG utilisé pour debug itératif

### Added
- `skeleton-quant-search/` — squelette prêt à copier pour le repo Quant-Search
- Règle **"Explain as if First Time"** : toujours expliquer IA, ML, concepts, maths comme si l'utilisateur ne savait rien (code en apprenant)
- Règle **"Sync with kuro-rules"** : toujours synchroniser les mises à jour de règles avec `~/Documents/kuro-rules`
