# NeuralDBG Ecosystem — Integration Contract

> MID: ECO-001
> Owner: LambdaSection
> Status: ACTIVE
> Last updated: 2026-06-07
> Source: PLAN.md Phase 11 (Bug-Solver Obligation) + Phase 3/6/7

## Vision

Trois dépôts distincts, **un seul pipeline** pour la crédibilité technique :

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

## The three repos

### 1. NeuralDBG (this repo) — Diagnostic Engine
- **Role** : instrumenter un training PyTorch, capturer les événements sémantiques, produire des hypothèses causales + JSON exportable.
- **Output contract** : `dbg.explain_failure()` → list[`CausalHypothesis`], `dbg.export_json()` → `events.json` (events, hypotheses, couplings, loss_history, first_failure)
- **Consumes** : `torch.nn.Module` (training loop)
- **Status** : v1.3.1 PyPI published, v1.3.2 in dev (FIX-001)

### 2. Aquarium (sibling repo, `~/Documents/Aquarium/`) — Visualizer
- **Role** : consommer le JSON exporté par NeuralDBG, le rendre sous forme d'arbre causal interactif (Tauri app, local).
- **Input contract** : `events.json` au format NeuralDBG (stable, versionné)
- **Output contract** : visualisation interactive + dashboard local
- **Status** : export JSON validé (`test_aquarium_export.py`), visualisation interactive MVP livrée

### 3. Neural-Agent (sibling repo, `~/Documents/Neural-Agent/`) — Auto-Corrector
- **Role** : consommer les `CausalHypothesis` de NeuralDBG, appliquer une règle de remédiation (LR, clip, init, mask fix…), patcher le training en live.
- **Input contract** : `dbg.explain_failure()` return value (Python objects, not JSON for in-process use)
- **Output contract** : `remediation_applied` event + patched optimizer/model state
- **Status** : prototype v0.x avec `RemediationRunner`, 5 règles (explosion, vanishing, dead neurons, saturation, data anomaly) — **M1 ajoute la règle MHA fully-masked-row** (issue BUG-001)

## Inter-repo contracts (stable)

### NeuralDBG → Neural-Agent (in-process, Python)
```python
from neuraldbg import NeuralDbg
from neuralagent import RemediationRunner

with NeuralDbg(model) as dbg:
    runner = RemediationRunner(dbg)  # wires to dbg.events live
    runner.register_default_rules()
    for step in range(N):
        loss = train_step(x, y)
        loss.backward()
        dbg.record_loss(loss.item())
        # runner auto-applies remediation if hypotheses match rules
```

Contract: `RemediationRunner` reads `dbg.events` and `dbg.explain_failure()` continuously. No shared state beyond `dbg`.

### NeuralDBG → Aquarium (out-of-process, JSON)
```bash
python train.py --neuraldbg-export events.json
# then open Aquarium app, drag events.json, view causal graph
```

Contract: `events.json` schema is stable and versioned. Schema in `neuraldbg/export_schema.json`. Backward-compat: Aquarium supports ≥ 2 prior versions.

### Neural-Agent → NeuralDBG (one-way dependency)
`neuralagent` depends on `neuraldbg` (PyPI). Not the other way around. `NeuralDBG` MUST remain usable without `neural-agent` installed.

## Roadmap integration points (Phase 11)

| Étape | NeuralDBG | Neural-Agent | Aquarium |
|-------|-----------|--------------|----------|
| M1 (Juin) | FIX-001 livré, v1.3.2 | Règle MHA wired | Lit BUG-001 JSON |
| M2 (mi-Juillet) | Benchmark 5+ scénarios | Publié PyPI, 10 règles | Dashboard bugs catalog |
| M3 (mi-Août) | 20+ bugs catalog | 25 règles, autonome | Catalogue visuel |

## Non-goals (M1-M3)

- ❌ Pas de SaaS cloud (Phase 7, post-M3)
- ❌ Pas de service payant
- ❌ Pas d'intégration closed-source tierce (W&B, MLflow) — on les compare, on ne les intègre pas
- ❌ Pas de modification de l'API publique NeuralDBG sans migration guide

## Open questions

- NeuralDBG expose-t-il un hook `on_hypothesis(h: CausalHypothesis)` pour pousser en async vers Aquarium live ? (à discuter en M2)
- Neural-Agent : règles stockées en YAML (git-friendly) ou Python (type-safe) ? Décision en M1.
- Aquarium : standalone desktop (Tauri) OU aussi web ? Standalone confirmé M1.
