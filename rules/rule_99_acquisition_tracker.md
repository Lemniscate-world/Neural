# RULE 99: Acquisition Tracker — Mémoire des Posts Marketing

## Trigger
Après CHAQUE action marketing : post Reddit, X, Discord, Lobsters, HN, blog, newsletter, etc.

## Obligation
Le fichier `docs/tracking/acquisition_tracker.md` DOIT être mis à jour dans les 5 minutes suivant chaque action.

## Format de logging

### Après un post publié
Ajouter une ligne dans la section "1. Posts publiés" :
```
| {DATE} | {Plateforme} | {Titre ou message (tronqué à 60 car)} | {Lien si public} | {Résultat: 🟢 positif / 🟡 en attente / 🔴 négatif} |
```

### Après un contact prospect
Ajouter dans "3. Contacts / Prospects" :
```
| {Nom/Handle} | {Plateforme} | {Date} | {Ce qui a été dit, intérêt exprimé} |
```

### Après chaque interaction notable
Ajouter dans "6. Leçons apprises" ce qui a marché ou non.

## Sanction
Si une session se termine sans que l'acquisition_tracker ait été mis à jour après une action marketing :
- L'agent DOIT le signaler comme tâche non complétée dans SESSION_SUMMARY.md
- Le post suivant DOIT être précédé d'une mise à jour du tracker

## Fichiers associés
- `docs/tracking/acquisition_tracker.md` — le tracker lui-même
- `LAUNCH_POSTS.md` — drafts des posts
- `docs/hn_feedback_log.md` — log HN