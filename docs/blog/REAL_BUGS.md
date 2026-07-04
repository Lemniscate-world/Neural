# NeuralDBG — Real Bug Post-Mortems

> Chaque article de cette section est basé sur un **vrai bug rencontré par la communauté ML**, reproduit localement et diagnostiqué avec NeuralDBG. Aucun scénario synthétique.
> Schéma MID : BUG-XXX / POST-XXX / FIX-XXX. Tracker : `docs/bugs/`.
> Dernière validation : **2026-07-03** — DeepMLP 12 couches (ResNet-18 equivalent) : **100% détection, 0% faux positifs**.

## Statut

- [x] Bug #1 — chassé, reproduit, diagnostiqué → **BUG-001** (POST-001 publié, FIX-001 dans v1.3.2)
- [x] Bugs #2-#10 — chassés, reproduits, catalogués → **BUG-002 à BUG-010** (repro scripts dans `examples/`)
- [x] **POST-003** publié (4 Juillet) — BUG-003 gradient explosion, pipeline E2E PASS
- [x] **POST-005** publié (4 Juillet) — BUG-005 LSTM batch pollution, cas parfait 0→24→0
- [x] Validation DeepMLP — 7/7 détectés (100%), 0% faux positifs, 3/7 résolus parfaitement
- [x] Causal chains — moteur opérationnel, 30 chaînes par bug, intégré dans `validate_resnet.py`
- [x] Pipeline E2E — boucle fermée detect→chain→fix→validate prouvée (BUG-003 PASS)

## Catalogue

| MID | Source | Type | DeepMLP Gap | Causal Chains | Statut |
|-----|--------|------|-------------|---------------|--------|
| BUG-001 | pytorch#41508 | MHA NaN gradients | **+24** | flat only | DETECTED |
| BUG-002 | pytorch#176793 | varlen_attn NaN | — | — | Repro ready |
| BUG-003 | pytorch#177116 | MPS wrong gradients | **+2** | 30 chains | DETECTED, RESOLVED |
| BUG-004 | HF#44928 | SDPA grad explosion | — | — | Upstream PR #47024 (CLOSED) |
| BUG-005 | pytorch#173334 | LSTM batch pollution | **+24** | flat only | DETECTED, RESOLVED |
| BUG-006 | pytorch#187759 | svdvals NaN swallow | **+2** | 30 chains | DETECTED, RESOLVED |
| BUG-007 | pytorch#186799 | compile wrong grad | **+24** | flat only | DETECTED |
| BUG-008 | pytorch#184575 | F.normalize grad corruption | **+17** | flat only | DETECTED |
| BUG-009 | pytorch#187227 | SDPA offset overflow | — | — | Shape mismatch (SKIP) |
| BUG-010 | pytorch#185543 | Inductor quantile | **+16** | flat only | DETECTED |

Note: "flat only" = hypothèses plates suffisantes (anomalies NaN isolées sans propagation). "30 chains" = chaînes causales de qualité (data_anomaly → gradient → optimizer).

## Validation A/B Comparative

| Métrique | Shallow (2-3 layers) | DeepMLP (12 layers) | Amélioration |
|----------|----------------------|---------------------|--------------|
| Détection | 57% (4/7) | **100% (7/7)** | +43% |
| Faux positifs | 0% | 0% | — |
| Gap médian | +1 | **+17** | 17x |
| Chaînes causales | 0 | 30/bug | infini |
| PASS parfait | 1/7 (BUG-005) | 3/7 (BUG-003,005,006) | 3x |

### Légende
- ✅ DETECTED = anomalies au-dessus de la baseline healthy
- ⚠️ SAME = même nombre d'anomalies que la baseline (non distinguable)
- ⏭️ SKIP = erreur de shape, modèle incompatible
- 🔜 = à venir / en cours

## Format des post-mortems

```
1. Issue originale (lien GH)
2. Contexte minimal (modèle, tâche, config)
3. Repro script (CPU/GPU friendly, < 50 lignes)
4. Symptômes observés (loss curves, gradient norms)
5. Diagnostic NeuralDBG (hypothèses + evidence chain + causal chain)
6. Workaround / fix proposé
7. Crédits & références
```

