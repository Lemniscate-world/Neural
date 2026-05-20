# Decision memo — NeuralDBG

**Date** : 2026-05-18
**Projet** : NeuralDBG
**Statut** : L1 ✅ — desk research complète (R75)

---

## Current decision

| Champ | Valeur |
|---|---|
| Verdict | **GO** — produit prêt, desk research complète, code fonctionnel |
| Geo priority | Global (US + Europe first) |
| Target user | ML engineers, researchers debugging PyTorch training |
| Buyer hypothesis | ML Platform Engineers ($50-200/user/mo), Foundation Model Trainers ($500+/mo) |
| Pain point | Debugging ML training is trial-and-error. No tool answers WHY a model failed. |

## Locked product definition

| Champ | Valeur |
|---|---|
| Wedge | Causal root cause analysis for PyTorch training — unique vs TensorBoard/W&B (WHY vs WHEN) |
| In scope | PyTorch hooks, semantic event extraction, causal hypotheses, Aquarium JSON export |
| Out of scope | Experiment tracking, hyperparameter sweeps, model registry, cloud hosting |

## Validation ladder

| Niveau | Statut | Preuve |
|---|---|---|
| L0 — problem hypothesis | ✅ | Desk research report (docs/desk_research_report.md) |
| L1 — desk evidence | ✅ | 4 personas with verbatim quotes, 8 competitors, TAM/SAM/SOM, 5 risks, 4 gaps |
| L2 — expert confirmation | 🔄 En cours | Lancement Show HN comme proxy d'expert calls |
| L3 — pilot-ready offer | ⏳ | Après Show HN |
| L4 — willingness-to-pay proof | ⏳ | Après pilot |

## Current evidence

| Ce qui est prouvé | Ce qui est partiellement prouvé | Ce qui est non prouvé |
|---|---|---|
| Pain point réel (desk research GO) | Volonté de payer (expert calls pas faits) | Prix acceptable |
| Produit fonctionnel (130 tests, PyPI) | Adoption organique (Show HN pas lancé) | Cycle de vente |
| Gap vs concurrents (causal chain unique) | Fit marché précis (wedge à valider) | Rétention |

## Decision rule

- [x] Desk research complète (L1) ✅ → autorise le lancement
- [ ] Expert calls (L2) pas encore faits → Show HN comme proxy
- [ ] Ne pas activer R77 (L2 Auto-Distribution Pipeline) tant que Show HN n'a pas confirmé le signal

## Next actions

1. **Lancer Show HN** (25 mai 2026) → valider L2 via feedback communautaire
2. Collecter signaux après HN → update evidence
3. Si signaux positifs → activer R77 (landing page + distribution pipeline)
4. Si signaux négatifs → documenter dans docs/launch_postmortem.md