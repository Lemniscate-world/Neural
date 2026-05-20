# Launch Plan — NeuralDBG

> Généré par R97 — Launch Planning Master Template
> Date : 2026-05-18
> Projet : NeuralDBG (v1.3.0-kuro)
> Statut : ✅ Desk research GO — produit prêt — lancement imminent
> Vérification : R98 — Pre-Launch MVP Verification Protocol

---

## Timeline

### T-7 à T-4 : 18-21 Mai — Vérifications R98 (Niveaux 1-2)

**Niveau 1 — Installation**
- [ ] `pip install neuraldbg` dans un venf frais
- [ ] `from neuraldbg import NeuralDbg` — import réussi
- [ ] Version dans pyproject.toml = 1.3.0 = CHANGELOG
- [ ] Dépendances OK (torch >= 2.0, psutil >= 5.9)

**Niveau 2 — Quickstart**
- [ ] Copier-coller le quickstart README → s'exécute sans erreur
- [ ] Le quickstart produit un résultat visible (events capturés, explications)
- [ ] Tester un scénario d'échec volontaire (vanishing avec petit LR)

**Général**
- [ ] Vérifier que GitHub Actions sont verts
- [ ] Vérifier que le README est à jour et lisible
- [ ] Vérifier la license MIT visible
- [ ] Mettre à jour la branche `main` avec les derniers commits
- [ ] Ajouter un badge PyPI dans le README

### T-3 : 21 Mai — Création des assets (fait)

- [x] Post Show HN dans `LAUNCH_POSTS.md`
- [x] X thread (R94) — 4 posts prêts
- [x] Post Reddit r/MachineLearning — prêt
- [x] Posts Discord (FrancophonIA, PyTorch, HF) — prêts
- [x] 7 réponses types aux questions HN probables

### T-1 : 24 Mai — Vérifications R98 (Niveaux 3-5)

**Niveau 3 — Tests fonctionnels**
- [ ] **Cas normal** : entraînement simple avec NeuralDbg → events capturés
- [ ] **Cas d'échec** : vanishing gradients → `explain_failure()` retourne des hypothèses
- [ ] **Export** : `export_aquarium_package()` → JSON valide
- [ ] **Mermaid** : `export_mermaid_causal_graph()` → graphe valide
- [ ] **Fallback** : `detect_coupled_failures()` sans engine → ne CRASH pas (retourne [])

**Niveau 4 — CI & Qualité**
- [ ] GitHub Actions verts
- [ ] pytest sans échec
- [ ] Badges README visibles (PyPI, license, Python, CI)
- [ ] Aucune issue critique ouverte

**Général**
- [ ] Tester le quickstart dans un venf frais
- [ ] Vérifier que le lien GitHub est bien accessible
- [ ] Vérifier que tous les posts sont formatés correctement
- [ ] Préparer l'alerte calendrier pour le jour J

**Niveau 5 — Final (soir J-1 ou matin J)**
- [ ] `pip install neuraldbg` dans un venf FRAIS
- [ ] Quickstart exécuté une dernière fois
- [ ] URL GitHub accessible
- [ ] README lisible sur mobile
- [ ] License bien affichée

### J : 26 Mai (mardi) — LANCEMENT 🚀

```
08:00 ET (14:00 FR) → Niveau 5 : vérification finale (venf frais, quickstart)
10:00 ET (16:00 FR) → POSTER SHOW HN
10:05 ET → Vérifier visibilité du post
10:10 ET → Auto-poster le premier commentaire HN
10:30 ET → Post X thread (4 posts)
11:00 ET → Post Reddit r/MachineLearning
12:00 ET → Post Discord (FrancophonIA + PyTorch + HF)
```

**Pourquoi le 26 mai (mardi) ?**
- Mardi/mercredi = meilleurs jours HN (25 mai est un lundi)
- 10h ET = 16h FR = pic de trafic US + Europe
- 1 semaine de préparation

### J+1 : 27 Mai — Suivi

- [ ] Logguer les métriques dans `docs/hn_feedback_log.md`
- [ ] Mettre à jour le README si nécessaire
- [ ] Remercier les contributeurs sur GitHub
- [ ] Répondre à TOUS les commentaires HN

### J+7 : 1 Juin — Amplification

- [ ] Écrire un blog technique (dev.to) : "How we built NeuralDBG"
- [ ] Si >100 upvotes HN : post "Show HN: What we learned from launching NeuralDBG"
- [ ] Si <20 upvotes : documenter les hypothèses d'échec dans `docs/launch_postmortem.md`

### J+30 : 25 Juin — Bilan

- [ ] Compiler les métriques : stars GitHub, téléchargements PyPI, emails
- [ ] Documenter les leçons dans `docs/launch_postmortem.md`
- [ ] Vérifier R69 : 3+ sources collectées à ce milestone
- [ ] Décider prochaine étape

---

## Check-list jour J (téléchargeable)

```markdown
☐ 08:00 — NIVEAU 5 : venf frais + pip install + quickstart
☐ 10:00 — POST SHOW HN sur news.ycombinator.com/submit
☐ 10:05 — Vérifier que le post est visible
☐ 10:10 — Auto-poster le commentaire HN
☐ 10:30 — X thread (4 posts)
☐ 11:00 — Reddit r/MachineLearning
☐ 12:00 — Discord FrancophonIA
☐ 12:05 — Discord PyTorch
☐ 12:10 — Discord Hugging Face
☐ 20:00 — Vérifier les commentaires de la journée
☐ 22:00 — Répondre aux commentaires en attente
```

---

## Script de vérification R98 (docs/verification_report_1.3.0_2026-05-18.md)

Le script de test complet est dans `rules/rule_98_prelaunch_verification.md` (template).
Exécuter avant J-1 pour produire le rapport.

---

## Ressources

| Ressource | Lien |
|---|---|
| Show HN submit | https://news.ycombinator.com/submit |
| X (Twitter) | https://x.com |
| Reddit r/MachineLearning | https://reddit.com/r/MachineLearning |
| Discord FrancophonIA | (salon #ia-general) |
| Discord PyTorch | (salon #showcase) |
| Discord Hugging Face | (salon #showcase) |
| GitHub | https://github.com/LambdaSection/NeuralDBG |
| PyPI | https://pypi.org/project/neuraldbg/ |

---

## Post-lancement

- [ ] Log HN feedback → `docs/hn_feedback_log.md`
- [ ] Mettre à jour `docs/tracking/acquisition_tracker.md` (PATH corrigé)
- [ ] Générer le X post quotidien (R94)
- [ ] Écrire SESSION_SUMMARY.md (R79)