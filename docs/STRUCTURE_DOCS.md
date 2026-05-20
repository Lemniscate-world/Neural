# R97 — STRUCTURE_DOCS : Template de Structure des Fichiers de Planification

**Trigger** : À la création de TOUT nouveau projet dans Kuro Ecosystem.
**Règle** : Chaque projet DOIT avoir ces fichiers, et un `docs/STRUCTURE_DOCS.md` qui les explique.

---

## Template universel

```markdown
# Structure des fichiers de planification — {NOM_DU_PROJET}

## À la racine

| Fichier | Contenu | Obligatoire |
|---|---|---|
| **PLAN.md** | Roadmap produit : phases, statut des features, vision écosystème | ✅ Oui |
| **decision-memo.md** | Décision marché : L0→L4, wedge, buyer hypothesis, validation ladder | ✅ Oui |
| **CHANGELOG.md** | Historique des versions : features ajoutées, bugs fixés | ✅ Oui |
| **LAUNCH_POSTS.md** | Contenu des posts : Show HN, X, Reddit, Discord, réponses types | ⏳ Si lancement public |
| **SESSION_SUMMARY.md** | Résumé de session : ce qui a été fait, ce qui reste | ✅ Oui (R79) |

## Dans docs/

| Fichier | Contenu |
|---|---|
| **docs/launch_plan_{PROJET}.md** | Planning temporel J-7 à J+30, check-list minute par minute |
| **docs/hn_feedback_log.md** | Log des retours HN après lancement |
| **docs/community_post_template.md** | Templates pour Reddit, Discord, X |
| **docs/launch_postmortem.md** | Analyse post-lancement (succès ou échec) |
| **docs/verification_report_*.md** | Rapport R98 : tests fonctionnels avant lancement |
| **docs/desk_research_report.md** | Recherche desk (personas, compétiteurs, TAM, risques, gaps) |
| **docs/failure_mode_table.md** | Tableau des risques techniques |

## Règle simple

- **PLAN.md** = "Quoi construire ?" (phases, features)
- **docs/launch_plan_{PROJET}.md** = "Quand et comment lancer ?" (timeline, actions)
- **decision-memo.md** = "Pourquoi et pour qui ?" (marché, wedge)
- **CHANGELOG.md** = "Qu'est-ce qui a été livré ?" (historique)
```

---

## Pour chaque nouveau projet

1. Copier ce template dans `docs/STRUCTURE_DOCS.md` du projet
2. Remplacer `{NOM_DU_PROJET}` et `{PROJET}` par le nom réel
3. S'assurer que tous les fichiers obligatoires (✅) existent
4. Ajouter les fichiers ⏳ si le projet est en phase de lancement
