---
name: Pull Request
about: Template de Pull Request obligatoire pour NeuralDBG
title: '[TYPE] Short description'
labels: 'Needs Review'
assignees: ''
---

## Description

<!-- Decrivez vos changements ici -->

## Rule 33 - Verification Checklist

**AVANT de soumettre cette PR, verifiez TOUS les points suivants:**

- [ ] **Je suis sur une branche `ceo/`** pour les modifications de regles OU sur `infra/`, `feat/`, `fix/` pour le code
- [ ] **Les regles (AGENTS.md, etc.) n'ont pas ete modifiees** sur cette branche (sauf si branche `ceo/`)
- [ ] **Cette branche est sync avec les dernieres regles** en provenance de la branche CEO
- [ ] **Aucun fichier protege** n'est inclus dans cette PR (voir .gitignore)

### Pour les branches non-CEO uniquement:

- [ ] J'ai merge les dernieres modifications de regles depuis la branche CEO
- [ ] Les regles locales sont identiques a celles de AGENTS.md sur main

### Verification des regles (Rule 15)

- [ ] Si j'ai modifie des regles, j'ai synchronise tous les fichiers de regles
- [ ] AGENTS.md, AI_GUIDELINESursorrules sont coher.md, .cents

## Rule 6 - Security Scan

- [ ] `bandit -r .` passe sans erreur
- [ ] `safety check` passe sans erreur

## Rule 5 - Test Coverage

- [ ] Tests unitaires ajoutes/modifies
- [ ] Couverture de code >= 60%
- [ ] Tous les tests passent en local

## Rule 30 - Branch Convention

- [ ] Nom de branche conforme: `scope/issue-id-description`
- [ ] Branche creee depuis la derniere version de main

## Linear Integration

- [ ] Tache Linear creee pour cette PR
- [ ] Numero de ticket Linear inclut dans le titre

## Changes Details

### Fichiers modifies:
-

### Tests ajoutes:
-

### Notes de deployment:
-

---
**Note:** Cette PR ne peut pas etre mergee sans validation complete de la checklist ci-dessus.
