# RULE 69: Intelligence Harvester — Collecte de Sources Externes

## Trigger
À chaque milestone de projet (L1 → L2, L2 → L3, avant tout lancement public).

## Objectif
Collecter 3+ sources externes (citations, études, posts, données) qui prouvent ou infirment une hypothèse clé du projet.

## Sources autorisées (par priorité)

| Priorité | Source | Type de preuve |
|---|---|---|
| 1 | Papers académiques (arXiv, ACL, NeurIPS) | Preuve forte, peer-reviewed |
| 2 | Données officielles (entreprises, régulateurs) | Preuve factuelle |
| 3 | Posts techniques (Reddit, HN, X, blogs) | Signal qualitatif |
| 4 | Études de marché (Gartner, Statista, rapports) | Preuve quantitative |
| 5 | Interviews / expert calls | Preuve primaire |

## Processus

1. Identifier l'hypothèse à vérifier (ex: "les devs ML galèrent avec le debugging de gradients")
2. Chercher 3+ sources qui y répondent
3. Pour chaque source, noter :
   - URL / citation exacte
   - Date de publication
   - Ce qu'elle prouve
   - Ce qu'elle ne prouve PAS
4. Ajouter les résultats dans le fichier evidence approprié (evidence-matrix.csv, desk_research_report.md)

## Emplacement des données

Les sources collectées sont stockées dans :
- `kuro-rules/KNOWLEDGE_BASE/mom_tests/` (si issues d'interviews)
- `kuro-rules/MARKETING_MEMORY/` (si issues de desk research)
- Le fichier `research/evidence-matrix.csv` du projet concerné

## Sanction

Si un milestone est passé sans 3+ sources collectées :
- Le projet est rétrogradé au niveau de validation précédent
- L'agent DOIT documenter pourquoi les sources manquent
- Aucun lancement public autorisé tant que les sources ne sont pas collectées

## Fichiers associés
- `research/evidence-matrix.csv` — matrice des preuves
- `research/scorecard.md` — scorecard du projet
- `docs/desk_research_report.md` — rapport de desk research