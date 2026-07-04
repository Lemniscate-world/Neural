# Changelog

All notable changes to NeuralDBG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (July 2026)
- **Causal chain engine** (`neuraldbg/causal_chain.py`): builds directed causal graphs from events, extracts ranked chains via DFS. Shows root cause → propagation → final symptom. Integrated via `dbg.explain_causal()`.
- **DeepMLP validation** (`validate_resnet.py`): 12-layer residual architecture achieving 100% detection (7/7) vs 57% on shallow models. Median gap: +17 anomalies.
- **GPU-trained Neural-Agent v3** (`train_balanced.py`): Qwen2-0.5B + LoRA, 5/5 categories distinct, trained on 10 live events + 30 real bug triplets.
- **End-to-end pipeline** (`e2e_pipeline.py`): detect → causal chain → AI diagnose → fix → validate. BUG-003 achieves PASS (0→24→1).
- **10 post-mortems published** on [GitHub Pages](https://lambdasection.github.io/NeuralDBG/blog/): complete catalog of real PyTorch/HF bugs with reproduction, diagnosis, and causal chains.
- **Validation dashboard** (`docs/dashboard.html`): live bug detection matrix, PR tracker, model versions.
- **Tool comparison matrix** (`docs/comparison.html`): NeuralDBG vs W&B/TensorBoard/MLflow/Captum across 16 capabilities. NeuralDBG: 14/16 YES.
- **Captum benchmark** (`benchmark_public/benchmark_captum.py`): proves NeuralDBG solves a different problem than explainability tools.
- **4 upstream PRs submitted**: #188933 (real fix), #188923 (+59/-0 test), #188053 (albanD reviewed), #188066 (CI fixed).
- **CLI wrapper**: `neuraldbg run script.py --agent --export`.
- **Live event capture** (`scripts/capture_live_events.py`): 10 live triplets from actual NeuralDBG sessions.
- **GPU agent bridge** (`agent_bridge.py` in Neural-Agent): subprocess-callable AI diagnosis.

### Changed
- **Agent model**: CPU distilgpt2 → GPU Qwen2-0.5B fp16 + LoRA (Quadro M4000, 8.6 GB).
- **PR creation**: moved from fork-clone to GitHub API direct (SHA-based) to avoid fork corruption.
- **Plan restructured**: focus on independent activities (content, product, distribution) while PRs await review.
- **Kaggle**: abandoned in favor of local GPU training.

### Fixed
- Causal chain engine: filter logic (AND/OR bug), node key collisions, DFS combinatorial explosion (45K→30 chains).
- PR #188066: 13 CI failures resolved (isnan→isfinite, TESTOWNERS header).
- PR #188797/#188922: closed corrupted PRs, replaced with clean #188923 (+59/-0).

## [1.3.2] - 2026-06-09

### Added
- **Multi-Repo Ecosystem cartography** (R105): NeuralDBG-Engine added as optional 4th component in [`docs/ecosystem.md`](docs/ecosystem.md); cross-repo SemVer tracking via new [`COMPATIBILITY_MATRIX.md`](COMPATIBILITY_MATRIX.md); "Écosystème (Multi-Repo)" section in `ROADMAP.md`.
- **Composite-module hook support**: `dbg.register_composite_hook(module)` for `nn.MultiheadAttention` and other modules with no leaf submodules.
- **Silent-loss and zero-leaf warnings**: detects loss=0 with non-zero gradients, and `register_full_backward_hook` no-op setups.
- **MHA fully-masked-row remediation rule**: `apply_mha_mask_workaround()` in Neural-Agent, wired to NeuralDBG events.
- **End-to-end Neural-Agent pipeline**: `diagnose -> fix -> validate -> apply -> re-run`, 87 tests passing.
- **Bug catalog BUG-001..004**: MHA NaN, varlen_attn NaN, MPS gradients, Qwen3.5 SDPA gradient explosion.
- **Public benchmark** (5 scenarios): all at 1.0 accuracy; comparison v2 vs W&B / MLflow / TensorBoard.
- **Aquarium JSON export**: full schema (`schema/events.json`), 14 unit tests in `test_aquarium_export.py`.
- **Phase 7 — Two-Package Architecture**: conditional import of `neuraldbg-engine` with seamless fallback in `neuraldbg` core.
- **Zero-Warnings Policy**: `filterwarnings` in `pyproject.toml` drops warnings 616 → 5.
- **Cross-repo contract**: `dbg.explain_failure()` and `events.json` schema v1 stable; `dbg` works without engine and without agent.

### Changed
- **PUBLIC → multi-repo narrative**: `ROADMAP.md` updated from "three-part" to "four-part" system (NeuralDBG, Neural-Agent, Aquarium, neuraldbg-engine).
- **Upstream PR tracker** updated: 4 comments posted, 1 PR submitted (pytorch/pytorch#186786, OPEN).
- **Benchmark table** expanded from 4 → 5 scenarios.

### Fixed
- Unicode/emoji terminal rendering encoding crash on Windows consoles for `quickstart.py`.
- Mock comparison removed from `benchmark_public/` — replaced by real `real_comparison.py` (R79 honesty).
- Deduplication of logical causal couplings in `detect_coupled_failures()` and Mermaid graph export.

### Security
- `assert` removed from production code paths (R39 compliance).
- Bandit scan wired to pre-commit (skips B101 — acceptable for tests).

## [1.3.1] - 2026-05-20

### Added
- **OOM Prevention & Memory Optimization**: Added `TensorDiskCache` to JIT-cache intermediate tensors on disk during anomaly states, preventing VRAM/RAM exhaustion.
- **Precision and Epsilon Scaling**: Implemented dtype-aware epsilon scaling (`1e-4` for float16/bfloat16, `1e-9` for float32/64) to prevent precision underflow during activation statistics computation.
- **Safety Guards for Integer Tensors**: Added strict checks (`torch.is_floating_point`) to bypass statistics computations on non-floating-point tensors (e.g., token indices, label masks), preventing PyTorch runtime errors.
- - Phase 2 dogfooding: LSTM/Time Series failure scenarios (vanishing recurrent, exploding recurrent, deep LSTM)
- - Phase 2 dogfooding: GNN (GCN/GAT) failure scenarios (oversmoothing, exploding, NaN injection)
- - Phase 2 dogfooding: torch.compile (Dynamo) compatibility scenarios (healthy, vanishing, exploding)
- - Phase 2 dogfooding: RL (PPO-style) failure scenarios (policy collapse, value explosion, reward hacking)
- - Phase 2 dogfooding: Distributed/DataParallel failure scenarios (healthy, vanishing, exploding under DP)
- - Engine fallbacks in core: `_classify_activation_health`, `explain_failure`, `detect_coupled_failures`, `export_mermaid_causal_graph`, `_classify_data_health`, `_check_data_anomaly` now work without proprietary engine
- - **Phase 3**: Complete Aquarium JSON export schema with all required fields (events, hypotheses, couplings, first_failure_layer, first_failure_step, loss_history)
- - 14 new unit tests for Aquarium export (`test_aquarium_export.py`)
- - Aquarium export integrated into LSTM demo with auto-export to `aquarium_exports/`
- - **Phase 7 — Two-Package Architecture**: Conditional import check for `neuraldbg-engine` and seamless fallback support for `neuraldbg` core, enabling private/public package separation.
- - **Zero-Warnings Policy**: Configured `filterwarnings` in `pyproject.toml` to ignore third-party deprecation warnings (MLflow, PyTorch full_backward_hook warnings), dropping warnings from 616 to 5.

### Fixed
- Fixed Unicode/emoji terminal rendering encoding crash on Windows consoles for `quickstart.py`.

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
