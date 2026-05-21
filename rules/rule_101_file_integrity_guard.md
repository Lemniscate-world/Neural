# RULE 101: File Integrity Guard — Protection des fichiers privés

## Trigger
Permanent — s'applique à CHAQUE opération git (commit, add, push).

## Protection Layers

### Layer 1 — .gitignore (automatique)
Les fichiers suivants sont en `.gitignore` et NE DOIVENT JAMAIS en être retirés :
```
decision-memo.md
LAUNCH_POSTS.md
docs/launch_plan_*.md
docs/hn_feedback_log.md
docs/community_post_template.md
docs/launch_postmortem.md
docs/cdp_protocol_definition.md
docs/STRUCTURE_DOCS.md
docs/verification_report_*.md
docs/tracking/
SESSION_SUMMARY.md
```

### Layer 2 — Pre-commit hook (automatique)
Le fichier `.githooks/pre-commit` bloque tout commit contenant ces fichiers.
Installé via `git config core.hooksPath .githooks`.

### Layer 3 — Vérification par l'IA (obligatoire)
Toute IA travaillant sur le projet DOIT :
1. Vérifier que `.githooks/pre-commit` existe et est actif
2. Vérifier que les patterns sensibles sont dans `.gitignore`
3. Ne JAMAIS proposer d'ajouter un fichier de planification au suivi git
4. Si un commit tente d'inclure ces fichiers → le bloquer et prévenir l'utilisateur

## Sanction
Si un fichier de planification est commité sur une branche distante (y compris par erreur) :
- L'agent DOIT le signaler comme violation dans SESSION_SUMMARY.md
- La correction DOIT être faite dans la même session : `git rm --cached` + `.gitignore` + `git commit --amend`

## Fichiers associés
- `.gitignore` — patterns de protection
- `.githooks/pre-commit` — hook de blocage
- `rules/COMPLIANCE_CHECKLIST.md` — R101 doit être ✅ DONE