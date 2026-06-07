# NeuralDBG — Bug Hunt Charter

> Stratégie de positionnement : faire de NeuralDBG le **Wireshark du ML** en prouvant la valeur par la résolution de bugs réels, pas par la fiction.

## Mission

Chasser des bugs d'entraînement ML **non résolus** sur des modèles **open-source**, les diagnostiquer avec NeuralDBG, et publier le résultat (article blog + commentaire sur l'issue originale).

## Critères de sélection d'une issue

| Critère | Requis | Pourquoi |
|---|---|---|
| **Statut** | Ouverte, pas de PR mergée | Pour apporter de la valeur |
| **Réproductible** | Sur CPU ou petit GPU (< 8GB VRAM) | Pas d'A100 dispo |
| **Modèle** | < 1B params, open-source | Téléchargeable rapidement |
| **Catégorie** | vanishing/exploding gradients, NaN loss, training collapse | Notre core competency |
| **Données** | Publiques (HF datasets, torchvision) | Repro transparente |
| **Engagement** | > 5 comments OU > 3 stars | Bug qui touche du monde |

## Repos prioritaires à scanner

- `pytorch/pytorch` (issues avec `grad` + `nan` + `training`)
- `huggingface/transformers` (Trainer bugs)
- `huggingface/diffusers` (training failures)
- `Lightning-AI/pytorch-lightning` (training bugs)
- `labmlai/annotated_deep_learning_paper_implementations`
- `tinygrad/tinygrad`
- `fastai/fastai`

## Process pour chaque bug

1. **Repro minimale** — script < 50 lignes qui déclenche le bug
2. **Wrap NeuralDBG** — ajouter 3 lignes de monitoring
3. **Capture le diagnostic** — `dbg.explain_failure()` + export JSON
4. **Véracité** — confirmer que l'hypothèse NeuralDBG pointe vers la bonne couche
5. **Workaround** — proposer un fix (même partiel)
6. **Publie** :
   - Article blog (`docs/blog/YYYY-MM-DD-<slug>.md` + `.html`)
   - Commentaire sur l'issue GH originale (lien vers l'article + repro script)
   - Tweet/thread X avec capture diagnostic

## KPIs

- 5 bugs résolus/analysés en 30 jours
- 3 articles blog publiés
- 5 issues GH où NeuralDBG est mentionné en solution
- 50+ étoiles GH gagnées via la chasse

## Garde-fous (Mom Test R2)

- **Ne jamais prétendre** avoir résolu un bug qu'on n'a pas reproduit
- **Toujours** fournir le script de repro et le log du diagnostic
- **Citer** l'issue originale et créditer le rapporteur
- **Préciser** si le diagnostic est partiel (workaround vs fix)

## Statut

- [x] Setup scan automatisé issues GH
- [x] Premier bug chassé et résolu → **BUG-001** (pytorch/pytorch#41508)
- [x] Premier article blog publié depuis un vrai cas → **POST-001** (2026-06-13)

## Catalogue en cours

| BUG | Source | Statut | Fix NeuralDBG | Article |
|-----|--------|--------|---------------|---------|
| BUG-001 | pytorch/pytorch#41508 (MHA NaN gradients) | Workaround confirmed, NeuralDBG fix in v1.3.2 (FIX-001) | `register_composite_hook()`, silent-loss warning, zero-leaf warning | POST-001 |
| BUG-002 | TBD (scan en cours) | À chasser | TBD | TBD |

