# NeuralSuite

> The complete toolkit for diagnosing and fixing deep learning training failures.

## Écosystème (Multi-Repo)

NeuralDBG fait partie d'un écosystème à 4 composants. Voir aussi :
- [docs/ecosystem.md](file:///c:/Users/Utilisateur/Documents/NeuralDBG/docs/ecosystem.md) — Contrat d'intégration (MID ECO-001)
- [COMPATIBILITY_MATRIX.md](file:///c:/Users/Utilisateur/Documents/NeuralDBG/COMPATIBILITY_MATRIX.md) — Matrice SemVer inter-repos

| Composant | Rôle | Statut |
|---|---|---|
| **NeuralDBG** (ce repo) | Moteur de diagnostic causal | v1.3.2 ✅ |
| **Neural-Agent** | Auto-correcteur | Pipeline built (closed beta) |
| **Aquarium** | Visualiseur IDE (Tauri) | MVP livré, dormant |
| **neuraldbg-engine** | Inférence causale avancée (optionnel) | v1.0.0 (registry privé) |

## What is NeuralSuite?

NeuralSuite is a four-part system that catches training problems before they waste your GPU hours:

| Component | What it does | Install |
|-----------|-------------|---------|
| **NeuralDBG** | Causal diagnostic engine — hooks into PyTorch, captures gradient/activation events, detects root causes | `pip install neuraldbg` |
| **Neural-Agent** | Auto-corrector — diagnoses failures and applies source-level fixes to training scripts | `pip install neural-agent` |
| **Aquarium** | Visualizer — interactive causal tree viewer for NeuralDBG exports | Desktop app (Tauri) |
| **neuraldbg-engine** *(optional)* | Advanced causal inference — adds data anomaly, optimizer instability, cross-arch coupling detection | Private registry (closed beta) |

## Why NeuralSuite?

Existing tools (W&B, MLflow, TensorBoard) are **dashboards** — they show you what happened. NeuralSuite tells you **why** it happened and **how to fix it**.

```
Training fails
    │
    ▼
NeuralDBG: "Vanishing gradients in layer_3, caused by Sigmoid saturation"
    │
    ▼
Neural-Agent: "Swap Sigmoid → LeakyReLU, increase LR by 2x"
    │
    ▼
Training runs successfully
```

## Quick Start

```python
from neuraldbg import NeuralDbg
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 64), nn.Sigmoid(), nn.Linear(64, 2))

with NeuralDbg(model) as dbg:
    for step in range(100):
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# See what went wrong
print(dbg.explain_failure())
```

## Roadmap

### v1.3.2 — Bug-Solver MVP (June 2026)
- [x] Composite-module hook support
- [x] Silent-loss and zero-leaf warnings
- [x] MHA fully-masked-row remediation rule
- [x] Neural-Agent: end-to-end diagnose -> fix -> validate pipeline
- [x] 300+ tests passing
- [x] Public benchmark: 4/4 scenarios at 1.0 accuracy
- [x] Tool comparison v2: NeuralDBG vs W&B vs MLflow vs TensorBoard
- [x] First upstream PR submitted (3 PRs!)
- [x] NeuralDBG-Engine: 45 tests, full API contract coverage
- [x] Neural-Agent: CPU training pipeline validated (distilgpt2 + LoRA, loss 4.18 -> 3.18)
- [x] **NEW: GPU training** — Qwen2-0.5B + LoRA on Quadro M4000 (8.6 GB), 5/5 categories
- [x] PR Gate system: mandatory 6-gate checklist before any upstream PR
- [x] **NEW: CLI Wrapper** — `neuraldbg run script.py` (zero-code injection)
- [x] **NEW: --agent flag** — Neural-Agent auto-fix after detection
- [x] **NEW: --export flag** — Aquarium JSON export

### Upstream PR Tracker

| Bug | Upstream Issue | PR Status | Merge Date |
|-----|---------------|-----------|------------|
| BUG-001 | pytorch/pytorch#41508 | Comment posted | - |
| BUG-002 | pytorch/pytorch#176793 | PR #186786 **CLOSED** (stale). New PR needed. | - |
| BUG-003 | pytorch/pytorch#177116 | **PR #188923 OPEN** (+59/-0, clean) | - |
| BUG-004 | huggingface/transformers#44928 | PR #47024 **CLOSED** (stale 1d). New PR needed. | - |
| BUG-005 | pytorch/pytorch#173334 | Repro posted, issue closed (awaiting reopen) | - |
| BUG-006 | pytorch/pytorch#187759 | **PR #188053 OPEN** — relance posted 3 Jul | - |
| BUG-007 | pytorch/pytorch#186799 | Cataloged (reported by @ezyang) | - |
| BUG-008 | pytorch/pytorch#184575 | **PR #188066 OPEN** — relance posted 3 Jul | - |
| BUG-009 | pytorch/pytorch#187227 | Cataloged (SDPA int32 overflow) | - |
| BUG-010 | pytorch/pytorch#185543 | Cataloged (quantile gradient mismatch) | - |

**PRs submitted**: 7 | **Active**: 4 (#188933 fix, #188923 test, #188053 test, #188066 test) | **Merged**: 0

### v1.4.5 — Catalog Expansion (July-August 2026)
- [x] 10/10 real bugs cataloged ✅ M2 OBJECTIVE REACHED
- [x] Reproducible public benchmark on 5+ real scenarios
- [ ] Comparison vs Captum (explainability)
- [x] 5/5 upstream PRs submitted
- [x] **Detection accuracy >= 0.90** → **1.00 (100%) on DeepMLP** ✅ EXCEEDED
- [x] **Causal chain engine** — true causal inference on computation graphs ✅
- [x] **GPU training** — Qwen2-0.5B + LoRA, balanced, 4/5 categories ✅

### v1.5.0 — Obligation (August-September 2026)
- [~] 10+ bugs cataloged ✅ | 10+ post-mortems published (3/10: POST-001, POST-003, POST-005)
- [ ] Versioned benchmark with CI regression gates
- [~] Neural-Agent autonomous: closed loop on ResNet + Transformer + GAN (DeepMLP done)
- [ ] 1+ upstream PR merged (external validation) — **0/5 merged, critical gap**
- [ ] Research paper draft on causal ML diagnostics

## Benchmark Results (v1.3.2)

5 scenarios, healthy excluded from averages:

| Tool | Detection (loss-only) | Detection (+grad norms) | Localization |
|------|:---------------------:|:-----------------------:|:------------:|
| **NeuralDBG** | **1.00** | **1.00** | **1.00** |
| W&B | 0.33 | 0.67 | 0.00 |
| MLflow | 0.33 | 0.67 | 0.00 |
| TensorBoard | 0.33 | 0.67 | 0.00 |

Key findings:
- External tools can detect anomalies (NaN loss, loss spikes) but cannot localize the failing layer
- With gradient norm logging, external tools detect MHA NaN gradients (0.33 -> 0.67)
- NeuralDBG is the only tool that names the layer causing the failure
- Benchmark is reproducible: `python -m benchmark_public.run`

## Install

```bash
pip install neuraldbg
```

## License

MIT
