# RULE 90: Livrables Mensuels — Mandatory Recall Protocol

## Purpose

Certains projets du portfolio lambda-Section ont des livrables récurrents mensuels
(contractuels, externes, ou engagements formels). Ces projets NE DOIVENT JAMAIS être
oubliés lors d'une session de travail, d'un bilan portfolio, ou d'une mise à jour
de Epingle_Projets.md.

## Trigger — OBLIGATOIRE

Cette règle s'active automatiquement dans les cas suivants :
- Au début de toute session touchant un projet listé ci-dessous
- Lors de toute mise à jour de Epingle_Projets.md (R80)
- Lors de tout bilan portfolio (R85 + R86)
- Lors de tout Discord/investor summary (R83)

## Projets avec Livrables Mensuels

| Projet | Section | Type de livrable | Fréquence | Partie prenante |
|--------|---------|------------------|-----------|-----------------|
| **OpenQuant** | λ-Section-2 | Modèle de trading (backtest validé) | ~1/2 mois (5/an) | Interne |
| **DevDemeterDAO** | Externe | Governance report / release | Mensuel | Demeter Labs |
| **Constant_Yield** | Externe | Protocol milestone / audit | Mensuel | Demeter Labs |
| **XP_Farming_System** | Externe | Contribution tracking release | Mensuel | Demeter Labs |
| **Hermes** | λ-Section-4 | Validation terrain update | Mensuel | Commercants Lomé |
| **Sagittarius** | λ-Section-9 | MLOps validation deliverable | Mensuel | Externe (B2B) |

## Comportement Obligatoire de l'Agent IA

```
LORS DE TOUTE SESSION :

1. CHECK : Les projets à livrables mensuels ci-dessus sont-ils concernés ?
2. IF oui :
   - RAPPELER la date approximative du prochain livrable
   - ALERTER si aucun SESSION_SUMMARY récent (< 14 jours) sur ce projet
   - DEMANDER à l'utilisateur si le livrable du mois est on-track
3. NEVER : Terminer une session sur un projet listé sans mentionner le statut du livrable mensuel
```

## Alerte de Retard

```
ALERT si :
  - Projet externe (Demeter Labs) sans commit depuis 14 jours → URGENT
  - OpenQuant sans nouveau modèle depuis 60 jours → RETARD sur objectif 5/an
  - Projet en validation sans mise à jour terrain depuis 21 jours → STALE
```

## Format de Rappel (à afficher en début de session concernée)

```
⚡ LIVRABLE MENSUEL DÉTECTÉ
Projet : [Nom]
Type : [Type de livrable]
Statut estimé : [On-track / Retard / Inconnu]
Dernier SESSION_SUMMARY : [Date ou "Introuvable"]
Action requise : [Préciser livrable ou confirmer avancement]
```

## Mise à Jour de Cette Règle

Quand un nouveau projet avec engagement mensuel est identifié :
1. Ajouter à la table ci-dessus
2. Mettre à jour la section `## ⚡ LIVRABLES MENSUELS` dans Epingle_Projets.md
3. Incrémenter la date de révision en bas de ce fichier

## Enforcement

**VIOLATION** : Terminer une session portfolio sans mentionner les projets à livrables mensuels.
**VIOLATION** : Mettre à jour Epingle_Projets.md sans vérifier le statut des livrables de ce mois.
**CORRECT** : Vérifier le tableau ci-dessus à chaque session, signaler tout retard.

## References

- R80 : Epingle_Projets.md maintenance
- R83 : Discord investor summary
- R84 : Validation pipeline automation
- R85 : Portfolio completeness verification
- R86 : Kuro Guardian surveillance
- R87 : Ownership intelligence (projets externes)

---

**Créé** : 2026-05-12
**Applique à** : Tous projets listés dans le tableau ci-dessus
**Enforcement** : MANDATORY — aucune exception
**Révision** : À mettre à jour si nouveaux engagements mensuels identifiés
