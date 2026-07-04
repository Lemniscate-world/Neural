# PLAN.md -- NeuralDBG Strategic Plan

> Last Updated: 2026-07-04 — **Repensé : focus sur l'indépendant, les PRs attendent.**

---

## 📊 Dashboard — 4 Juillet 2026

| Pilier | Status | Dépend de nous ? |
|--------|--------|:-----------------:|
| **Détection** | 🟢 100% | ✅ Oui |
| **Causal Chains** | 🟢 Opérationnel | ✅ Oui |
| **GPU Model** | 🟢 v3 (5/5) | ✅ Oui |
| **Pipeline E2E** | 🟢 Prouvé | ✅ Oui |
| **Blog** | 🟡 3/10 posts | ✅ Oui |
| **PRs** | 🔴 4 actives, 0 merges | ❌ Non (maintainers) |
| **Stars** | 🔴 24 | ❌ Non (communauté) |

**Principe** : Investir 80% du temps sur ce qu'on contrôle. Les PRs et les stars suivront.

---

## 🎯 Plan Révisé — Ce qu'on fait MAINTENANT (sans dépendre des PRs)

### Bloc A : Contenu & Crédibilité (contrôlable à 100%)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| A1 | **7 post-mortems restants** → 10/10 | Blog riche = crédibilité | 2h/pièce |
| A2 | **Benchmark vs Captum** | Comparaison académique | 4h |
| A3 | **Vidéo démo 3 min** | "NeuralDBG in action" | 3h |
| A4 | **1 bug par jour sur X/Reddit** | Visibilité quotidienne | 30min/j |

### Bloc B : Produit (contrôlable à 100%)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| B1 | **Dashboards de validation** (HTML interactif) | Preuve visuelle | 3h |
| B2 | **Améliorer chaînes causales** (filtrer bruit) | Qualité diagnostic | 2h |
| B3 | **Réentraîner modèle** (10→30 live events) | Précision agent | 1h GPU |
| B4 | **Tests sur modèles réels** (ResNet, GPT-2) | Coverage | 4h | ✅ **FAIT** |

### Bloc C : Distribution (contrôlable à 80%)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| C1 | **PyTorch Dev Discussions** (poster analyses) | Visibilité maintainers | 1h |
| C2 | **Reddit r/MachineLearning** (trouver proxy) | Visibilité large | 1h |
| C3 | **Hacker News** (build karma → Show HN) | Visibilité tech | Continu |

### Bloc D : PRs (en attente — suivi passif)

| # | Action | Fréquence |
|---|--------|-----------|
| D1 | Relance commentaires | Tous les 3-4 jours |
| D2 | Vérifier statut "actionable" | 1x/semaine |
| D3 | Répondre aux reviews (si elles arrivent) | Immédiat |

---

## ⏱️ Timeline Réaliste

| Semaine | Focus | Livrables |
|---------|-------|-----------|
| **7-11 Juillet** | Bloc A (contenu) | 5 post-mortems, benchmark Captum |
| **14-18 Juillet** | Bloc B (produit) | Dashboard, chaînes améliorées, modèle v4 |
| **21-25 Juillet** | Bloc C (distribution) | Posts communautaires, Discussions |
| **Août+** | Bloc D (PRs) + itération | Suivi passif, répondre aux reviews |

---

## État des lieux — 4 Juillet 2026

### Formation GPU: v3 finale

| Version | Date | LR | Catégories | Live events | Loss |
|---------|------|-----|-----------|-------------|------|
| v1 (biased) | 3 Jul | 2e-4 | 1/5 (saturated only) | 6 | 0.0002 |
| v2 (balanced) | 3 Jul | 5e-5 | 4/5 | 6 | 0.001 |
| **v3 (enriched)** | **4 Jul** | **5e-5** | **5/5 distinctes** | **10** | **0.0004** |

### Validation A/B

| Métrique | Shallow | DeepMLP | Cible |
|----------|---------|---------|-------|
| Détection | 57% | **100%** | ≥90% ✅ |
| Faux positifs | 0% | 0% | 0% ✅ |
| Gap médian | +1 | **+17** | >+5 ✅ |
| PASS parfait | 1/7 | 3/7 | — |
| Causal chains | 0 (buggé) | 2/7 (qualité) | — |

### Validation Architectures Réelles — 4 Juillet

| Architecture | Bugs détectés | FP | Détails |
|-------------|---------------|-----|--------|
| Mini ResNet (CNN, 4 blocks) | 4/5 (80%) | 0/1 | Miss: sigmoid saturation (subtle) |
| Mini Transformer (3 encoders) | **5/5 (100%)** | 0/1 | Tous les bugs capturés |
| **Total** | **9/10 (90%)** | **0/2** | Zero false positives |

### PR Upstream — Statut 4 Juillet

| PR | Bug | Type | Statut | Reviews | Issue |
|----|-----|------|--------|---------|-------|
| #188933 | BUG-002 varlen_attn | **Real fix** | OPEN | 0 | Actionable demandé |
| #188923 | BUG-003 MPS | Test (+59/-0) | OPEN | CI bot | Actionable demandé |
| #188053 | BUG-006 svdvals | Test | OPEN | **1 (albanD)** | Process expliqué |
| #188066 | BUG-008 F.normalize | Test | OPEN | CI fixé | Actionable demandé |

**Leçon albanD**: Discuter sur l'issue AVANT la PR. Faire marquer "actionable". Suivre le process PyTorch.

### Blog / Post-mortems

| Post | Bug | Contenu | Status |
|------|-----|---------|--------|
| POST-001 | BUG-001 MHA NaN | HTML, pas de chain | Publié (13 Juin) |
| POST-003 | BUG-003 Gradient Explosion | HTML + Mermaid + SVG charts | **Publié (4 Juillet)** |
| POST-005 | BUG-005 LSTM Batch Pollution | HTML + Mermaid + SVG charts | **Publié (4 Juillet)** |
| POST-00x | BUG-006, 008, 010 | À écrire | TODO |

### Livraisons 3-4 Juillet

| Date | Livrable |
|------|----------|
| 3 Jul | Causal chain engine fix (3 bugs) |
| 3 Jul | DeepMLP validation (100% detection) |
| 3 Jul | GPU v2 + v3 training |
| 3 Jul | PR crisis: #188797/#188922 closed, #188923 created clean |
| 3 Jul | PR relances #188053, #188066 |
| 3 Jul | PR #188933 (BUG-002 real fix) created |
| 4 Jul | Pipeline E2E proven (BUG-003 PASS) |
| 4 Jul | albanD review → process PyTorch appris, 4 issues postées |
| 4 Jul | #188066 CI fix (isnan→isfinite + lint) |
| 4 Jul | Blog: POST-003 + POST-005 HTML + Mermaid + SVG charts |
| 4 Jul | 10 live events capturés (BUG-001,003,005,006,008,010) |

1. ~~CLA PyTorch non signée~~ → **✅ SIGNÉE (24 Juin). PR #186786 mergeable.**
2. **HN low karma** → impossible de poster Show HN
3. **r/MachineLearning compte restreint** → pas de reach Reddit large
4. **PR #186786: 0 reviews en 16 jours** → dépend des mainteneurs PyTorch

---

## 📚 R89 — Leçons Apprises (PRs upstream)

> Ajouté le 24 Juin 2026 après analyse des échecs.

### Ce qui a ÉCHOUÉ

| PR | Approche | Résultat | Leçon |
|----|----------|----------|-------|
| **#186631** (BUG-001, pytorch#41508) | `warnings.warn()` dans MHA quand NaN détecté | **Fermée par le CEO.** Les mainteneurs PyTorch ne mergeront jamais un simple warning. | **D2: Un warning n'est PAS un fix.** PyTorch attend une correction du comportement, pas une détection. |
| **BUG-001 v2** (post-186631) | Full pipeline NeuralDBG+Agent: détecter → diagnostiquer → corriger | **Pas encore transformé en PR.** Le pipeline est prêt mais la PR upstream n'a pas été créée. | **Le pipeline sans PR = 0 crédibilité.** NeuralDBG+Agent doit aboutir à une PR concrète. |
| **PR #186786** (BUG-002) | `ValueError` explicite dans `varlen_attn()` | **OPEN depuis 16 jours, 0 reviews.** | **Une bonne PR ne suffit pas.** Sans visibilité (commentaires, @mentions, partage), elle reste invisible. |

### Ce qui a MARCHÉ

| Action | Résultat | Leçon |
|--------|----------|-------|
| Commentaires sur les issues upstream | 4/4 postés, visibilité établie | **Commenter AVANT de coder.** Établir la présence d'abord. |
| Repro scripts indépendants du hardware | BUG-003/004 testés sur CPU via injection | **Tester sans GPU.** Les mainteneurs peuvent reproduire. |
| PR #186786: fix réel (pas warning) | Validation d'entrée qui prévient le bug | **Un vrai fix > un warning.** Même si pas encore mergé, c'est le bon type de contribution. |

### Règles pour les PROCHAINES PRs upstream

1. **Pas de `warnings.warn`.** Soit on corrige le comportement, soit on ne fait pas de PR.
2. **Toujours inclure un repro script** qui tourne sur CPU (ou avec fallback).
3. **Toujours citer NeuralDBG** dans le message de commit ou la description PR.
4. **Après avoir soumis, partager** : PyTorch Dev Discussions, X, Reddit, Discord.
5. **Suivre la PR** : si pas de review après 1 semaine, commenter poliment.

---

## Vision

NeuralDBG est le **moteur de diagnostic** pour l'entraînement de réseaux de neurones.
Aquarium en est le **visualiseur/IDE**.
Neural-Agent est l'**agent auto-correcteur** qui utilise les diagnostics causaux pour
corriger automatiquement un entraînement qui échoue.

> La valeur n'est pas le code, c'est le **format machine-readable** que seul cet outil produit.
> NeuralDBG devient le protocole structuré (causal chains, event types, root causes)
> qu'un agent IA consomme pour auto-diagnostiquer et auto-réparer un training.

## Écosystème

| Repo | Rôle | Statut | PyPI |
|------|------|--------|------|
| `NeuralDBG/` | Moteur d'instrumentation & diagnostic causal | v1.3.2, 5 scénarios benchmark | `pip install neuraldbg` ✅ |
| `Neural-Agent/` | Agent auto-correcteur (diagnose + fix) — **PRIVE, PAYANT** | Pipeline built, modèle pas entraîné | **Privé — pas sur PyPI public** |
| `Aquarium/` | IDE visuel (Tauri) | Dormant | N/A |
| `NeuralDBG-Engine/` | Moteur causal propriétaire (optionnel) | v1.0.0 packagé | **Privé — GitHub Packages / private registry** |

### Dépendances inter-repos

```
neuraldbg-engine  ──(optionnel)──▶  neuraldbg  ◀──(requis)──  neural-agent
                                       │
                                       │ events.json
                                       ▼
                                   aquarium
```

- `neuraldbg-engine` est chargé dynamiquement par `neuraldbg` (import conditionnel). Sans lui, fallbacks.
- `neural-agent` consomme `neuraldbg.explain_failure()` en in-process.
- `aquarium` lit les exports JSON de `neuraldbg` (out-of-process).

### Artefacts écosystème (R105/R106)

| Fichier | Rôle | État |
|---|---|---|
| `PLAN.md` (ce fichier, privé gitignored) | Plan stratégique interne | ✅ à jour |
| `ROADMAP.md` (public) | Roadmap public + quickstart | ✅ à jour |
| `docs/ecosystem.md` (MID ECO-001, public) | Contrat d'intégration inter-composants | ✅ à jour (4 composants) |
| `COMPATIBILITY_MATRIX.md` (racine, public) | Matrice SemVer inter-repos | ✅ créé (2026-06-09) |
| `docs/architecture/INFERENCE_FLOW.md` | Flux d'inférence causal | ✅ |
| `docs/cdp_protocol_definition.md` | Contrat sans/avec engine | ✅ |

---

## Priorités Immédiates (Semaine du 24 Juin)

> Mise à jour après le run live du 24 Juin. Les statuts reflètent la réalité.

| # | Action | Pourquoi | Effort | Qui | Statut |
|---|--------|----------|--------|-----|--------|
| **P1** | **Signer CLA pytorch#186786** | Bloque merge PR #186786 depuis 16 jours | 5min | CEO | **🔴 BLOQUÉ** |
| **P2** | **Créer PR upstream BUG-003 (pytorch#177116)** | Objectif M2: 3 PRs. On en a 1. | 2h | AI | **TODO** |
| **P3** | **Créer PR upstream BUG-004 (hf#44928)** | Objectif M2: 3 PRs. | 2h | AI | **TODO** |
| **P4** | **Créer PR upstream BUG-005 (pytorch#173334)** | 5ème bug → PR = crédibilité | 2h | AI | **TODO** |
| **P5** | **Merger PR #665 (ecosystem-cartography)** | 26 fichiers, +2524 lignes en attente | 5min | CEO | **✅ FAIT (24 Juin)** |
| **P6** | **Chasser 5 bugs supplementaires** | Objectif M2: 10 bugs. On en a 5. | 1w | AI | **TODO** |
| **P7** | **Lancer entraînement Kaggle Neural-Agent** | Pipeline sans modèle = 0 valeur | 30min | CEO | **✅ FAIT — Pipeline CPU validé (24 Juin)** |
| **P8** | **Ajouter suite de tests NeuralDBG-Engine** | 0 tests aujourd'hui = risque régression | 3h | AI | **✅ FAIT — 45 tests créés (24 Juin)** |

---

## Breaches / Erreurs detectees (24 Juin)

| # | Probleme | Impact | Resolution |
|---|----------|--------|------------|
| 1 | Aucun commentaire postee sur les 4 issues upstream | Zero visibilite upstream | **RESOLU** — 4 commentaires postés via gh CLI |
| 2 | Zero PR upstream soumises | Pas de preuve de contribution | **RESOLU** — PR #186786 soumise (pytorch varlen_attn) |
| 3 | GitHub Pages non active | Landing + blog invisibles | ~~CEO: Settings > Pages~~ RESOLU ✅ |
| 4 | r/MachineLearning bloque | Compte restreint | Trouver un proxy ou commenter |
| 5 | HN bloque | Low karma | Commenter des posts tech pour build karma |
| 6 | BUG-003 documente mais pas de draft commentaire | Commentaire pas pret | RESOLU ✅ (draft cree) |
| 7 | Acquisition tracker pas mis a jour depuis 7 juin | Historique incomplet | Mettre a jour apres chaque post |
| 8 | pytorch PR #186631 ferme par CEO | PR naive (warnings.warn) sans pipeline NeuralDBG+Agent | BUG-004 montre le full pipeline: detect+fix |
| **9** | **PR #186786 ouvert depuis 16 jours, 0 reviews** | **Merge rate = 0%**. Sans CLA signée, la PR est invisible. | **CEO: signer CLA (5 min)** |
| **10** | **NeuralDBG-Engine: 0 tests** | **Aucune suite de tests**. Le package privé n'a aucun filet de sécurité. | **✅ RESOLU — 45 tests créés (24 Juin)** |
| **11** | **PR #665 (ecosystem-cartography) non mergée** | **26 fichiers, +2524 lignes** en suspens sur `feat/ecosystem-cartography`. | **✅ RESOLU — Mergé dans main (24 Juin)** |
| **12** | **Neural-Agent: modèle jamais entraîné** | Pipeline codé (87 tests) mais le modèle est un squelette. | **🟡 PARTIEL — Pipeline CPU validé. Kaggle GPU reste TODO.** |

---

## Phases Complétées

### Phase 1 — Stabilisation & Hardening (v1.3.0) ✅
- Governance & Rules
- Infrastructure (Makefile, bootstrap, bump_version)
- Core Engine : coupling dedup, nommage modules, tests qualité
- ResNet-18 demo (vanishing, exploding, data anomaly)
- 105+ tests pass, version taggée

### Phase 1.5 — Audit Remediation & Core Stabilization (v1.3.1) ✅
- Hygiène des dépendances
- Refonte IDs Causaux (UUID stricts)
- OOM Prevention & Disk Cache (TensorDiskCache)
- Zero Warnings Policy (616 → 5 warnings)
- Unicode Windows Compatibility
- Type Safety sur Tenseurs

### Phase 0 — Validité Causale — Moteur Infaillible ✅
- Validité causale : NaN dans couche X → engine localise X
- Faux positifs : entraînement sain → 0 hypothèses
- Déterminisme : même seed + même bug → mêmes hypothèses
- Mutation : N modes de défaillance → N détectés
- Scalabilité : 1000 modules → hooks < 1s
- Benchmark causal → accuracy 0.917 (grid search)

### Phase 2 — Dogfooding Extensif ✅
- Transformer (nanoGPT), GANs (DCGAN), LLM fine-tuning (LoRA)
- Diffusion (DDPM), Distributed/DataParallel, LSTM/Time Series
- GNN (GCN/GAT), RL (PPO/DQN), torch.compile

### Phase 4 — Desk Research (R75) ✅
- Personas (4), Competitors (8), Market Sizing (TAM $16B)
- Risk Analysis (5), Gap Analysis (4), GO Decision

### Phase 5 — Publication PyPI ✅
- Package publié : neuraldbg v1.3.1 sur PyPI
- Workflow CI/CD publish.yml actif

---

## Phases Actives

### Phase 10 — MVP Launch & Go-to-Market (18 Mai → 25 Juin)
> **Statut : FIN DE PHASE.** Le 25 Juin est demain. Bilan mitigé.

#### Métriques réelles vs objectifs (J+37 au 24 Juin)

| Métrique | Objectif J+30 | Réel (24 Juin) | Écart |
|----------|---------------|------|-------|
| GitHub stars | 50 | 24 | -52% |
| PyPI downloads/mois | 100 | 161 (au 8 Juin) | ✅ |
| HN post | Show HN | BLOQUÉ (low karma) | ❌ |
| Qualified feedback | 5 | 1 | -80% |
| Issues/PRs community | 3+ | 0 | -100% |
| PR upstream mergée | 1 | 0 | ❌ |

#### Posts publiés
| Date | Plateforme | Résultat |
|------|-----------|----------|
| 2026-05-19 | Reddit r/deeplearning | 982 vues, 1 commentaire qualifié |
| 2026-05-19 | Discord FrancophonIA | Sans réponse |
| 2026-05-19 | Discord PyTorch | Sans réponse |
| 2026-05-26 | X / Twitter | Posté |
| 2026-05-26 | HackerNews | BLOQUÉ |
| 2026-05-29 | Reddit r/PyTorch | Posté, en attente |
| 2026-05-30 | GitHub Discussion #664 | Posté |
| 2026-06-07 | PyTorch Forums | Posté |

#### Posts à publier (CEO TODO — drafts prets)
| Plateforme | Draft | Statut |
|------------|-------|--------|
| pytorch/pytorch#41508 (BUG-001) | `docs/posts/pytorch_41508_comment.md` | **POSTE** |
| pytorch/pytorch#176793 (BUG-002) | `docs/posts/pytorch_176793_comment.md` | **POSTE** |
| pytorch/pytorch#177116 (BUG-003) | `docs/posts/pytorch_177116_comment.md` | **POSTE** |
| huggingface/transformers#44928 (BUG-004) | `docs/posts/huggingface_44928_comment.md` | **POSTE** |
| r/MachineLearning | `docs/posts/reddit_ml_draft.md` | **BLOQUE (compte restreint)** |

#### PRs upstream à soumettre (CRITIQUE — mis à jour 24 Juin)
| Issue | Bug | Repro script | PR | Statut |
|-------|-----|-------------|-----|--------|
| pytorch/pytorch#41508 | BUG-001 MHA NaN | `examples/repro_pytorch_41508.py` ✅ | N/A (CEO: warnings-only PR fermé) | **PR naive fermée. Leçons apprises (R89)** |
| pytorch/pytorch#176793 | BUG-002 varlen_attn NaN | `examples/repro_pytorch_176793.py` ✅ | [#186786](https://github.com/pytorch/pytorch/pull/186786) ✅ | **OPEN depuis 16 jours — 0 reviews, CLA non signée** |
| pytorch/pytorch#177116 | BUG-003 MPS gradients | `tests/unit/test_mps_gradient_detection.py` ✅ | A CREER | **TODO** |
| huggingface/transformers#44928 | BUG-004 Qwen3.5 SDPA | `examples/repro_huggingface_44928.py` ✅ | A CREER (comment posté) | **TODO** |
| pytorch/pytorch#173334 | BUG-005 LSTM batch pollution | `examples/repro_pytorch_173334.py` ✅ | A CREER (draft commentaire prêt) | **TODO** |

#### Assets créés ✅
- README "Killer" avec comparatif visuel
- Landing Page (GitHub Pages)
- Blog index (GitHub Pages)
- 2 post-mortems HTML/MD
- `examples/` — 3 scripts de reproduction (pytorch_41508, pytorch_176793, huggingface_44928)
- `notebooks/train_neuralagent_kaggle.ipynb`
- `.github/PR_TEMPLATES/upstream-fix.md`
- Benchmark 5 scénarios + comparison v2
- BUG-004: Qwen3.5 SDPA gradient explosion (HuggingFace downstream)
- `docs/posts/huggingface_44928_comment.md` — upstream comment draft

---

### Phase 11 — Bug-Solver Obligation (3 mois → v1.4.0 → v1.5.0)
> **Objectif** : passer de "diagnostic tool" à "obligation de résolution".
> Sur tout bug OSS chassé, on détecte, on résout, on mesure, on compare.
> Schéma MID : BUG-XXX / POST-XXX / FIX-XXX / DEC-XXX

#### M1 — v1.4.0 "Bug-Solver MVP" (Juin → mi-Juillet)

| Tâche | Statut | Notes |
|-------|--------|-------|
| BUG-001 chassé : pytorch/pytorch#41508 | ✅ | MHA NaN gradients |
| POST-001 publié | ✅ | Post-mortem complet |
| FIX-001 livré | ✅ | `register_composite_hook()`, silent-loss, zero-leaf warnings |
| 9 tests FIX-001 | ✅ | `tests/unit/test_composite_hook.py` |
| Bug-hunt charter + REAL_BUGS | ✅ | Mis à jour |
| Plan écosystème | ✅ | `docs/ecosystem.md` |
| Neural-Agent règle MHA wired | ✅ | `remediation_rules.py` + `apply_mha_mask_workaround()` |
| Agent pipeline complet | ✅ | agent.py, predict.py, rewriter, sandbox, validator, CLI |
| 87 tests Neural-Agent | ✅ | Tous passent |
| Benchmark public 5 scénarios | ✅ | 5/5 at 1.0, comparison v2 |
| Comparaison vs W&B/MLflow/TensorBoard | ✅ | NeuralDBG 1.0/1.0/1.0 vs 0.50/0.75/0.00 |
| Mock comparison supprimé (honesty) | ✅ | Replaced by real_comparison.py |
| BUG-002 chassé : pytorch/pytorch#176793 | ✅ | varlen_attn NaN gradients with padding |
| BUG-003 chassé : pytorch/pytorch#177116 | ✅ | MPS catastrophically wrong gradients |
| Blog index (GitHub Pages) | ✅ | `docs/blog/index.html` |
| Kaggle notebook entraînement | ✅ | `notebooks/train_neuralagent_kaggle.ipynb` |
| PR template upstream | ✅ | `.github/PR_TEMPLATES/upstream-fix.md` |
| Landing page redesign | ✅ | Logo supprimé, bleu abysses, bouton Blog |
| **Commentaire pytorch#41508 (BUG-001)** | **POSTE** | https://github.com/pytorch/pytorch/issues/41508#issuecomment-4659336137 |
| **Commentaire pytorch#176793 (BUG-002)** | **POSTE** | https://github.com/pytorch/pytorch/issues/176793#issuecomment-4659337880 |
| **Commentaire pytorch#177116 (BUG-003)** | **POSTE** | https://github.com/pytorch/pytorch/issues/177116#issuecomment-4659340137 |
| **Commentaire huggingface/transformers#44928 (BUG-004)** | **POSTE** | https://github.com/huggingface/transformers/issues/44928#issuecomment-4659341957 |
| **Activer GitHub Pages** | **RESOLU** | Active par CEO |
| **1ere PR upstream soumise** | **FAIT** | PR #186786 — varlen_attn NaN (pytorch#176793) |
| **Cartographie écosystème (R105/R106)** | **FAIT** | `docs/ecosystem.md` (4 composants) + `COMPATIBILITY_MATRIX.md` (nouveau) + section ROADMAP |
| **Cross-repo integration tests** | **FAIT (squelette fonctionnel)** | `tests/integration/cross_repo/` (3 fichiers, 15 tests, mark `cross_repo`) |
| **Commit v1.3.2 + tag** | **PARTIEL** | Tag `v1.3.2` ✅ existe. CHANGELOG v1.3.2 ✅ fait sur `feat/ecosystem-cartography` (commit `73ed24d6`). Release commit sur `main` à faire (CEO TODO) |

#### M2 — v1.4.5 "Catalog Expansion" (mi-Juillet → mi-Août)

- [ ] 10 bugs chassés (MHA, GNN, LSTM, GAN, diffusion, transformers x2, RL, LoRA, FSDP)
  - BUG-001: pytorch MHA NaN ✅
  - BUG-002: pytorch varlen_attn NaN ✅
  - BUG-003: pytorch MPS gradients ✅
  - BUG-004: HuggingFace Qwen3.5 SDPA gradient explosion ✅
  - (6 more needed)
- [ ] Benchmark causal public sur 5+ scénarios
- [ ] Comparaison vs W&B / MLflow / TensorBoard / Captum
- [ ] **Neural-Agent publié sur PyPRIVÉ** (BLOQUE: pas encore entraîné, PRIVATE/PAYANT)
- [ ] Aquarium interopérable
- [ ] **3+ upstream PRs soumises** (1/3 — PR #186786 soumise, CLA en attente)
- [ ] **3+ commentaires postés sur issues upstream** (4/4 ✅ — bugs 001-004 tous postés)
- [ ] Accuracy causale ≥ 0.90

#### M3 — v1.5.0 "Obligation" (mi-Août → mi-Septembre)

- [ ] 20+ bugs, 10+ post-mortems
- [ ] Benchmark causal versionné avec seuils CI
- [ ] Comparaison vs 4 outils sur 5 bugs communs
- [ ] Neural-Agent autonome : boucle fermée ResNet + Transformer + GAN
- [ ] 1+ PR upstream mergée
- [ ] Research paper draft
- [ ] Plan SaaS Cloud

#### Conditions de succès

| Métrique | M1 | M2 | M3 |
|----------|----|----|----|
| Bugs catalog | 3 | 10 | 20 |
| Post-mortems | 1 | 8 | 12 |
| FIX-XXX | 1 | 5 | 10 |
| Commentaires upstream postés | **4** | 6 | 12 |
| PRs upstream soumises | **1** | 3 | 6 |
| PRs upstream mergées | 0 | 1 | 2 |
| Benchmark accuracy | 1.0 | 0.90 | 0.93 |
| Comparaisons publiées | 1 | 1 | 3 |
| Packages PyPI | 1 (neuraldbg) | 1 | 1 |

---

### Phase 6 — Agent Auto-Correcteur (Neural-Agent)
> **Statut : Pipeline built, modèle pas entraîné, PAS SUR PYPI.**
> Repo : `~/Documents/Neural-Agent/`

#### Ce qui est fait ✅
- [x] Créer repo Neural-Agent
- [x] Prototype boucle fermée (demo_remediation.py)
- [x] Règle MHA fully-masked-row wired to NeuralDBG events
- [x] Agent pipeline complet : agent.py → diagnose → fix → validate → apply → re-run
- [x] Inference engine (model/predict.py) : HF transformers + GGUF backends
- [x] Code rewriter (code/rewriter.py) : grad clipping, LR, activation swap, batch norm, MHA
- [x] Safe execution sandbox (code/sandbox.py)
- [x] Fix validator (validator.py)
- [x] CLI (cli.py) : `neuralagent fix/diagnose/collect/train/export`
- [x] Triplet collector (data/collector.py) : 6 catégories, 110 triplets en 27s
- [x] JSONL formatter (data/format.py)
- [x] 87 tests passent

#### Ce qui reste à faire (BLOQUE)
- [ ] **Publier neural-agent** (privé, payant — distribution via canal commercial)
- [ ] **Faire tourner `neuralagent collect` + `neuralagent train`** (GPU requis → Kaggle)
- [ ] Prouver que le fine-tuné surpasse le dictionnaire de règles
- [ ] Consommer `dbg.explain_failure()` directement (contrat inter-repo)
- [ ] Boucle fermée robuste sur architectures réelles (ResNet, Transformer)
- [ ] 25 règles de remédiation

---

## Phases Futures (pas encore démarrées)

### Phase 3 — Aquarium (Visualisateur Graphique)
- Repo existant (`~/Documents/Aquarium/`), dormant
- Export JSON validé, visualisation SVG/Canvas de base
- **Reprendre quand le diagnostic est mature**

### Phase 7 — Stratégie IP, Business Model & SaaS Cloud
- [ ] `neuraldbg-engine` sur registry privée (GitHub Packages)
- [ ] Déposer la marque (INPI/USPTO)
- [ ] API REST `/diagnose` (events JSON → diagnostic propriétaire)
- [ ] SaaS Aquarium Cloud

### Phase 8 — Auto-Improvement Causal (Bayesian Tuning)
- [x] Grid search initial : accuracy 0.917
- [ ] Bayesian Search des seuils de divergence
- [ ] Validation cross-benchmarks (GitHub Issues, HuggingFace, Kaggle)

### Phase 9 — Auto-Validation Continue & Régression CI
- [ ] Validation accuracy minimale à chaque commit
- [ ] Alerte de régression si accuracy chute > 0.05

---

## Garde-fous Mom Test (R2 / R64)

- Ne jamais prétendre résoudre un bug non reproduit localement
- Ne jamais citer un benchmark sans script de repro public
- Ne jamais revendiquer une comparaison sans même code testé
- Toujours fournir repro + diagnostic log + workaround/fix
- **Ne jamais marquer "fait" quand c'est un draft non posté**
- **Ne jamais oublier de publier sur PyPI avant de dire "disponible"**
- **Ne jamais confondre "publié sur GitHub" et "posté sur la plateforme cible"**

## Cross-references

- `docs/ecosystem.md` — contrat d'intégration NeuralDBG / Aquarium / Neural-Agent
- `docs/bugs/BUG-XXX-*.md` — catalogue de bugs OSS chassés
- `docs/decisions/DEC-XXX-*.md` — décisions archivées
- `docs/tracking/acquisition_tracker.md` — historique posts marketing
- `docs/blog/YYYY-MM-DD-POST-XXX-*.md` — post-mortems publics
- `docs/posts/*_draft.md` — drafts prets a poster
- `docs/posts/*_comment.md` — commentaires prets a poster sur issues upstream
- `benchmark_public/` — benchmark causal reproductible
