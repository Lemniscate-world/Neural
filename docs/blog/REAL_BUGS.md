# NeuralDBG — Real Bug Post-Mortems

> Chaque article de cette section est basé sur un **vrai bug rencontré par la communauté ML**, reproduit localement et diagnostiqué avec NeuralDBG. Aucun scénario synthétique.
> Schéma MID : BUG-XXX / POST-XXX / FIX-XXX. Tracker : `docs/bugs/`.

## Statut

- [x] Bug #1 — chassé, reproduit, diagnostiqué → **BUG-001** (POST-001 publié, FIX-001 dans v1.3.2)
- [ ] Bug #2 — idem
- [ ] Bug #3 — idem

## Catalogue

| MID | Source | Article | Fix NeuralDBG | Repro | Statut |
|-----|--------|---------|---------------|-------|--------|
| BUG-001 | pytorch/pytorch#41508 | POST-001 (2026-06-13) | FIX-001 / v1.3.2 | examples/repro_pytorch_41508.py | Workaround confirmed, fix in v1.3.2 |
| BUG-002 | TBD | TBD | TBD | TBD | Scan en cours |

## Format des post-mortems

```
1. Issue originale (lien GH)
2. Contexte minimal (modèle, tâche, config)
3. Repro script (CPU/GPU friendly, < 50 lignes)
4. Symptômes observés (loss curves, gradient norms)
5. Diagnostic NeuralDBG (hypothèses + evidence chain)
6. Workaround / fix proposé
7. Crédits & références
```

