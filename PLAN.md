# PLAN.md — Tactical Execution

## Vision
NeuralDBG est le **moteur de diagnostic** pour l'entraînement de réseaux de neurones.
Aquarium en est le **visualiseur/IDE**.
Le prochain palier est un **agent auto-correcteur** qui utilise les diagnostics causaux pour
corriger automatiquement un entraînement qui échoue.

> Dans l'ère des agents IA qui codent tout, une librairie Python seule n'a pas de sens.
> NeuralDBG devient le **protocole structuré** (causal chains, event types, root causes)
> qu'un agent IA consomme pour auto-diagnostiquer et auto-réparer un training.
> La valeur n'est pas le code, c'est le **format machine-readable** que seul cet outil produit.

## Écosystème
- `NeuralDBG/` → Moteur d'instrumentation & diagnostic causal ← **ici**
- `Aquarium/` → IDE visuel (Tauri) dans `~/Documents/Aquarium/`
- `Neural-Agent/` → Agent auto-correcteur ← **à créer**
- `Neural-Research/` → Recherche amont (dans Documents)
- `Neural-Again/` → Itération architecture (dans Documents)

## Chronological Roadmap

### Phase 1 : Stabilisation & Hardening (DONE — v1.3.0)
- [x] Governance & Rules
- [x] Infrastructure (Makefile, bootstrap, bump_version)
- [x] Core Engine : coupling dedup, nommage modules, tests qualité
- [x] ResNet-18 demo (vanishing, exploding, data anomaly)
- [x] 105+ tests pass, version taggée

### Phase 1.5 : Audit Remediation & Core Stabilization (DONE — v1.3.1)
> **Objectif** : Résoudre la dette technique, les risques OOM et éliminer la pollution de logs.

| Tâche | Statut | Commentaire / Solution |
|---|---|---|
| **Hygiène des dépendances** | ✅ | Purge des failles `.venv` (`pytest` upgradé). |
| **Refonte IDs Causaux** | ✅ | Remplacement du mapping string par des hash UUID stricts (`uuid.uuid4().hex`). |
| **OOM Prevention & Disk Cache** | ✅ | Implémentation du `TensorDiskCache` pour décharger les tenseurs de diagnostic hors de la VRAM. |
| **Zero Warnings Policy** | ✅ | Ajout des filtres PyTest dans `pyproject.toml` (warnings descendus de 616 à 5). |
| **Unicode Windows Compatibility** | ✅ | Retrait des emojis unicode de `quickstart.py` pour éviter les plantages Windows CP1252. |
| **Type Safety sur Tenseurs** | ✅ | Guardage strict via `torch.is_floating_point` pour ignorer les types entiers (tokens/masks). |

### Phase 0 : Validité Causale — Moteur Infaillible (DONE)
| Test | Statut |
|---|---|
| **Validité causale** : NaN dans couche X → engine localise X | ✅ |
| **Faux positifs** : entraînement sain → 0 hypothèses | ✅ |
| **Déterminisme** : même seed + même bug → mêmes hypothèses | ✅ |
| **Mutation** : N modes de défaillance → N détectés | ✅ |
| **Scalabilité** : 1000 modules → hooks < 1s | ✅ |
| **API Contract** : export JSON valide | ✅ |
| **Invariance cross-architecture** : MLP + ResNet + Transformer | ✅ |
| **Régression CI** : seuils d'hypothèses stables | ✅ |
| **Benchmark causal** → accuracy 0.917 (grid search) | ✅ |

### Phase 2 : Dogfooding Extensif (DONE)
| Architecture | Type de défaillance | Statut |
|---|---|---|
| **Transformer** (nanoGPT) | Attention collapse, NaN softmax, LR warmup | ✅ |
| **GANs** (DCGAN generator) | Vanishing, exploding, NaN injection | ✅ |
| **LLM fine-tuning** (LoRA) | Catastrophic forgetting, loss spikes, NaN | ✅ |
| **Diffusion** (DDPM) | NaN UNet, exploding gradients, noise collapse | ✅ |
| **Distributed/DataParallel** | Multi-GPU hook integrity | ✅ |
| **LSTM/Time Series** | Vanishing recurrent gradients | ✅ |
| **GNN** (GCN/GAT) | Oversmoothing, deep GNN | ✅ |
| **RL** (PPO/DQN) | Reward hacking, policy collapse | ✅ |
| **torch.compile** | Dynamo graph compatibility | ✅ |

### Phase 4 : Desk Research (R75) — MANDATORY (DONE)
- [x] Personas (4 personas avec verbatim Reddit/HN/SO)
- [x] Competitors (8 : W&B, Neptune, MLflow, TensorBoard, Captum, Comet, WhyLabs, OpenAI Clarity)
- [x] Market Sizing (TAM $16B, SAM $1.2B, SOM $2M/3yr)
- [x] Risk Analysis (5 risques avec probabilité/impact/remède)
- [x] Gap Analysis (4 gaps avec preuves)
- [x] **GO Decision** — toutes les 5 dimensions complétées

### Phase 5 : Publication PyPI (DONE)
- [x] Mettre à jour `pyproject.toml` (metadata, auteurs, keywords, classifiers)
- [x] Créer `.github/workflows/publish.yml` (TestPyPI + PyPI)
- [x] Build local : `neuraldbg-1.3.1.tar.gz` + `neuraldbg-1.3.1-py3-none-any.whl`
- [x] Twine check : PASSED
- [x] Test install fresh venv : `pip install ./dist/neuraldbg-1.3.1-py3-none-any.whl`
- [x] Environments GitHub configurés (testpypi, pypi) via workflow publish.yml
- [x] Package publié sur PyPI : version 1.3.0 disponible
- [x] Tag v1.3.0 créé et pushé sur origin
- [x] Workflow auto-publish déclenché sur release GitHub

### Phase 10 : MVP Launch & Go-to-Market (COURANTE : 18 Mai - 25 Juin)
> **Objectif** : Exécuter le plan de lancement public, acquérir les 100 premiers utilisateurs qualifiés et collecter des feedbacks techniques.
> **Calendrier Détaillé** : Se référer au plan de lancement officiel [launch_plan_neuraldbg.md](file:///c:/Users/Utilisateur/Documents/NeuralDBG/docs/launch_plan_neuraldbg.md).

#### 1. Métriques de Succès (KPIs de Validation à J+30)
| Métrique | Objectif | Pourquoi |
|---|---|---|
| **Installs PyPI** | 100 | Preuve d'utilisabilité et d'installation fonctionnelle |
| **GitHub Stars** | 50 | Preuve d'intérêt et de crédibilité technique |
| **Retours Qualifiés** | 5 | Preuve que l'outil résout un vrai problème de debug |
| **Issues/PRs** | 3+ | Preuve d'engagement de la communauté open-source |

#### 2. Cibler les Early Adopters (Personas Phase 4)
- [ ] **Le PhD Student** (Reddit r/MachineLearning, r/deeplearning)
- [ ] **Le ML Engineer** (HackerNews, Twitter/X)
- [ ] **Le Researcher** (Twitter/X, Discord IA)

#### 3. Assets de Lancement MVP
- [x] **README "Killer"** : Section "Why NeuralDBG?" avec comparatif visuel.
- [x] **Landing Page** (GitHub Pages) : Page statique dans `docs/index.html`.
- [x] **Démo Vidéo** : Script `scripts/record_demo.py` pour enregistrement automatisé.
- [x] **Exemple "Copy-Paste"** : `quickstart.py` à la racine (robuste sur Windows).

#### 4. Statut des Canaux de Distribution & Pré-lancement (R98)
- [x] **Niveau 1 (Installation)** : `pip install` local wheel dans venv frais + import OK + version 1.3.1 cohérente.
- [x] **Niveau 2 (Quickstart)** : Exécution de `quickstart.py` OK (sans crash d'encodage, rapports générés).
- [ ] **HackerNews (Show HN)** : Draft rédigé dans `LAUNCH_POSTS.md`. Prévu le mardi 26 mai, 10h EST.
- [ ] **Reddit** : Post prêt dans `LAUNCH_POSTS.md` pour r/MachineLearning.
- [ ] **Twitter/X** : Thread technique prêt dans `LAUNCH_POSTS.md`.
- [ ] **Discord** : Annonces prêtes pour les serveurs PyTorch, HF et FrancophonIA.

---

## Post-Launch Roadmap (v1.4.0+ & SaaS Cloud)

### Phase 3 : Pipeline Aquarium (Visualisateur Graphique)
> **Objectif** : Interfaçer les diagnostics causaux exportés avec une application visuelle interactive locale.

- [x] **Setup repo Aquarium** (GitHub) : `~/Documents/Aquarium/`
- [x] **Export JSON** : Structure complète (events, hypotheses, couplings, loss_history, first_failure) validée par `test_aquarium_export.py`.
- [ ] **Visualisation interactive** : Rendu interactif en d3.js/Mermaid au sein de l'application Tauri.
- [ ] **Tableau de bord local** : Interface temps réel pour analyser les steps d'entraînement en cours.

### Phase 6 : Agent Auto-Correcteur (Auto-Remediation)
> **Objectif** : Intégrer les diagnostics machine-readable avec un agent autonome capable de modifier les paramètres d'entraînement à chaud.

- [x] Créer repo `~/Documents/Neural-Agent/`
- [ ] Protocole d'auto-remédiation : Sortie causal NeuralDBG $\rightarrow$ Action d'ajustement.
- [ ] Agent correcteur : Écriture de scripts pour intercepter les exceptions, appeler `explain_failure()` et modifier le LR, la taille de batch ou l'initialisation de couches.
- [ ] Boucle fermée automatisée : Entraînement $\rightarrow$ Crash $\rightarrow$ Diagnostic causal $\rightarrow$ Correction autonome $\rightarrow$ Reprise de l'entraînement.

### Phase 7 : Stratégie IP, Business Model & SaaS Cloud
> **Objectif** : Protéger notre propriété intellectuelle (moat) tout en construisant le modèle commercial.

- **Actions Propriétaires** :
  - [ ] Publier `neuraldbg-engine` sur une registry privée (GitHub Packages).
  - [ ] Déposer la marque "NeuralDBG" (INPI/USPTO).
  - [ ] Commercialisation Cloud : API REST (POST `/diagnose` prenant les events JSON $\rightarrow$ retournant le diagnostic propriétaire).
  - [ ] Hébergement SaaS d'Aquarium Cloud et centralisation de l'historique d'entraînement.

### Phase 8 : Auto-Improvement Causal (Bayesian Tuning)
- [x] Grid search initial : Amélioration de l'accuracy causale à **0.917**.
- [ ] Optimisation continue : Bayesian Search automatique des seuils de divergence (`threshold_vanishing`, `threshold_exploding`, saturation thresholds).
- [ ] Validation cross-benchmarks (GitHub PyTorch Issues, HuggingFace forums, Kaggle error logs).

### Phase 9 : Auto-Validation Continue & Régression CI
- [ ] Intégration dans le workflow CI : Validation de l'accuracy causale minimale à chaque commit sur le benchmark d'anomalies.
- [ ] Alerte de régression : Détecter et stopper les commits si l'accuracy globale du moteur chute de > 0.05.

---
**Last Updated**: 2026-05-20
