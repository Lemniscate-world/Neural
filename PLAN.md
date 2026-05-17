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

## Roadmap

### Phase 1 : Stabilisation & Hardening (DONE — v1.3.0)
- [x] Governance & Rules
- [x] Infrastructure (Makefile, bootstrap, bump_version)
- [x] Core Engine : coupling dedup, nommage modules, tests qualité
- [x] ResNet-18 demo (vanishing, exploding, data anomaly)
- [x] 105+ tests pass, version taggée

### Phase 0 : Validité Causale — Moteur Infaillible ✅
| Priorité | Test | Statut |
|----------|------|--------|
| 🔴 **Critique** | **Validité causale** : NaN dans couche X → engine localise X | ✅ |
| 🔴 **Critique** | **Faux positifs** : entraînement sain → 0 hypothèses | ✅ |
| 🔴 **Critique** | **Déterminisme** : même seed + même bug → mêmes hypothèses | ✅ |
| 🟡 **Haut** | **Mutation** : N modes de défaillance → N détectés | ✅ |
| 🟡 **Haut** | **Scalabilité** : 1000 modules → hooks < 1s | ✅ |
| 🟡 **Haut** | **API Contract** : export JSON valide | ✅ |
| 🟢 **Moyen** | **Invariance cross-architecture** : MLP + ResNet + Transformer | ✅ |
| 🟢 **Moyen** | **Régression CI** : seuils d'hypothèses stables | ✅ |
| 🆕 | **Benchmark causal** → accuracy 0.917 (grid search) | ✅ |

### Phase 2 : Dogfooding Extensif
| Priorité | Architecture | Type de défaillance | Statut |
|----------|-------------|---------------------|--------|
| 🔴 Haute | **Transformer** (nanoGPT) | Attention collapse, NaN softmax, LR warmup | ✅ |
| 🔴 Haute | **GANs** (DCGAN generator) | Vanishing, exploding, NaN injection | ✅ |
| 🟡 Moyenne | **LLM fine-tuning** (LoRA) | Catastrophic forgetting, loss spikes, NaN | ✅ |
| 🟡 Moyenne | **Diffusion** (DDPM) | NaN UNet, exploding gradients, noise collapse | ✅ |
| 🟡 Moyenne | **Distributed/DataParallel** | Multi-GPU hook integrity | ✅ |
| 🟢 Basse | **LSTM/Time Series** | Vanishing recurrent gradients | ✅ |
| 🟢 Basse | **GNN** (GCN/GAT) | Oversmoothing, deep GNN | ✅ |
| 🟢 Basse | **RL** (PPO/DQN) | Reward hacking, policy collapse | ✅ |
| 🟢 Basse | **torch.compile** | Dynamo graph compatibility | ✅ |

### Phase 3 : Pipeline Aquarium (connexion au repo Aquarium)
| Priorité | Tâche | Statut |
|----------|-------|--------|
| 🔴 Haute | **Setup repo Aquarium** (GitHub) | ❌ |
| 🔴 Haute | **Export JSON** vers Aquarium — schéma complet (events, hypotheses, couplings, loss_history, first_failure) | ✅ |
| 🟡 Moyenne | **Visualisation** des causal graphs | ❌ |
| 🟢 Basse | **Dashboard** temps réel | ❌ |

**Schéma JSON Aquarium :**
```json
{
  "events": [{"type", "layer", "step", "from", "to", "confidence", "metadata"}],
  "hypotheses": [{"description", "confidence", "causal_chain"}],
  "couplings": [...],
  "first_failure_layer": {...},
  "first_failure_step": {...},
  "loss_history": [...]
}
```

**Tests :** 14 tests unitaires (`test_aquarium_export.py`) + intégration dans les demos
- [ ] Itérer sur le format si nécessaire

### Phase 4 : Desk Research (R75) — MANDATORY ✅
- [x] Personas (4 personas avec verbatim Reddit/HN/SO)
- [x] Competitors (8 : W&B, Neptune, MLflow, TensorBoard, Captum, Comet, WhyLabs, OpenAI Clarity)
- [x] Market Sizing (TAM $16B, SAM $1.2B, SOM $2M/3yr)
- [x] Risk Analysis (5 risques avec probabilité/impact/remède)
- [x] Gap Analysis (4 gaps avec preuves)
- ✅ **GO Decision** — toutes les 5 dimensions complétées

### Phase 5 : Publication PyPI
- [x] Mettre à jour `pyproject.toml` (metadata, auteurs, keywords, classifiers)
- [x] Créer `.github/workflows/publish.yml` (TestPyPI + PyPI)
- [x] Build local : `neuraldbg-1.3.0.tar.gz` + `neuraldbg-1.3.0-py3-none-any.whl`
- [x] Twine check : PASSED
- [x] Test install fresh venv : `pip install ./dist/neuraldbg-1.3.0-py3-none-any.whl` ✅
- [ ] Configurer environments GitHub (testpypi, pypi) avec trusted publishing
- [ ] Publier sur TestPyPI pour validation
- [ ] Tag v1.3.0 + release GitHub → auto-publish PyPI

### Phase 6 : Agent Auto-Correcteur
- [x] Créer repo `~/Documents/Neural-Agent/`
- [ ] Définir le protocole : NeuralDBG causal output → action
- [ ] Implémenter un agent qui reçoit `explain_failure()` → ajuste LR/init/archi
- [ ] Boucle fermée : training → diagnostic → correction → nouveau training

### Phase 7 : Stratégie IP & Business Model

#### Constat
> GPT-5.5 et les agents IA avancés peuvent copier du code MIT et reproduire des tests
> en 24h. Le code open source est un *vecteur de distribution*, pas un *moat*.
> Si on reste purement MIT, un compétiteur (ou un agent) peut cloner, exécuter,
> et livrer plus vite que nous.

#### Décision Stratégique
**Hybride : Core MIT + Engine Propriétaire + Cloud SaaS**

| Couche | Licence | Contenu | Pourquoi |
|--------|---------|---------|----------|
| 🆓 **Core** | MIT (public) | Hooks PyTorch, collecte d'events, contexte `NeuralDbg`, démos basiques | Distribution, adoption, écosystème |
| 🔒 **Engine** | Propriétaire (privé) | Raisonnement causal avancé, heuristiques de couplage, patterns de défaillance, base de connaissances | Le vrai moat — pas copiable |
| ☁️ **Cloud** | SaaS (privé) | API REST, file d'attente de diagnostics, auto-correction, Aquarium-hosted, historique | Revenue, lock-in data |
| 🤖 **Agent** | Propriétaire (privé) | Boucle fermée training → diag → correction, fine-tuning par feedback | Automatisation, escalabilité |

#### Actions Concrètes

##### Protection IP (immédiat)
- [ ] **Déposer la marque "NeuralDBG"** (INPI / USPTO) — protection du nom
- [ ] **Copyright explicite** sur la suite de tests (`tests/`) — les tests sont l'ADN du moteur
- [ ] **Secret commercial** pour les heuristiques de couplage causal (dans le code privé)
- [ ] **Dépôt logiciel** (APP / LDAP) — preuve d'antériorité

##### Restructuration du Code — Stratégie Two-Package ✅

**Architecture des paquets :**

```
┌──────────────────────────────────────────────┐
│  neuraldbg  (public — PyPI — MIT)            │
│  pip install neuraldbg                       │
│  Repo: github.com/NeuralDBG/neuraldbg        │
│                                              │
│  - Hooks PyTorch (forward/backward)          │
│  - SemanticEvent, CausalHypothesis           │
│  - Context manager NeuralDbg                 │
│  - Export JSON Aquarium                      │
│  - Fallbacks heuristiques (sans engine)      │
│  - Demos basiques                            │
│                                              │
│  try: import neuraldbg_engine                │
│  except ImportError: pass  ← optionnel       │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  neuraldbg-engine  (privé — registry privée) │
│  pip install neuraldbg-engine                │
│  Repo: ~/Documents/NeuralDBG-Engine          │
│                                              │
│  - Classification gradients/activations      │
│  - Inférence causale avancée                 │
│  - Détection d'anomalies de données          │
│  - Couplages de défaillances                 │
│  - Hypothèses causales détaillées            │
│  - Base de patterns de défaillance           │
└──────────────────────────────────────────────┘
```

- [x] Repo `neuraldbg-engine/` existe (`~/Documents/NeuralDBG-Engine`) avec structure complète
- [x] Import conditionnel dans `neuraldbg/__init__.py` (try/except)
- [x] Fallbacks heuristiques implémentés quand engine absent
- [x] 130 tests passent avec engine, fallbacks fonctionnent sans engine
- [ ] Publier `neuraldbg` sur PyPI public
- [ ] Publier `neuraldbg-engine` sur registry privée (GitHub Packages)
- [ ] Documenter : `pip install neuraldbg` = gratuit, `pip install neuraldbg-engine` = payant

##### Cloud SaaS
- [ ] Définir l'API : POST `/diagnose` (events JSON) → `{ root_causes, hypotheses, couplings }`
- [ ] Déployer un endpoint serverless (AWS Lambda / Cloudflare Workers)
- [ ] Free tier : diagnostic basique (3 scénarios par jour)
- [ ] Pro tier : illimité + agent auto-correcteur + Aquarium Cloud
- [ ] Enterprise : on-prem engine, audit trail, SLA

##### Business Model
| Tier | Prix | Cible |
|------|------|-------|
| 🆓 Core (MIT) | Gratuit | Indie researchers, hobbyists |
| ☁️ Cloud Free | Gratuit (limité) | ML engineers en exploration |
| ☁️ Cloud Pro | $29/mo | Individuels, petites équipes |
| ☁️ Cloud Team | $99/mo | Équipes ML (5-20 users) |
| 🏢 Enterprise | Sur devis | Grands comptes, finance, santé |

#### Ce que même GPT-5.5 ne peut pas copier
| Actif | Pourquoi c'est protégé |
|-------|----------------------|
| **Engine causal propriétaire** | Code jamais publié, jamais exposé |
| **Base de patterns de défaillance** | Accumulée par dogfooding sur des architectures réelles (plus précieuse que le code) |
| **Infrastructure cloud** | Données clients, historique de diagnostics, feedback loops |
| **Marque "NeuralDBG"** | Protection légale, notoriété |
| **Aquarium** | IDE Tauri séparé, écosystème propre |
| **Tests de validité causale** | Privés, non reproductibles sans connaissance du domaine |

### Phase 8 : Auto-Improvement Causal — ML sur le ML
Le benchmark causal + grid search a déjà amélioré l'accuracy de 0.750 → 0.917.
Prochaine étape : faire du vrai ML sur le moteur de diagnostic.

#### Pipeline d'auto-amélioration
```
Scenario Generator → Dataset causal → Grid/Bayesian Search → Meilleurs seuils
      ↑                                                    ↓
      └────── Validation sur benchmarks externes ←─────────┘
```

#### Dataset causal (NeuralDBG-Engine/benchmark/scenarios.py)
| Scénario | Bug | Architecture | Vérité terrain |
|----------|-----|-------------|---------------|
| Vanishing | Weights zeroed | MLP 3 couches | layer "3", step 2 |
| Exploding | Weights ×1000 | MLP 3 couches | layer "0", step 2 |
| NaN injection | NaN dans weight | MLP 3 couches | layer "2", step 3 |
| Healthy | Aucun | Linear 1 couche | 0 hypothèses |
| Dead Neurons | Poids → -100 | MLP + ReLU | layer "1", step 2 |
| GAN Exploding | Poids ×500 | Generator | layer "0", step 2 |
| Attention collapse | Embedding ×1e6 | Transformer | layer "wte", step 2 |
| à ajouter | Catastrophic forgetting | LoRA | ? |

#### Hyperparamètres du moteur causal (à optimiser)
| Paramètre | Valeur défaut | Meilleur trouvé | Impact |
|-----------|--------------|-----------------|--------|
| `threshold_vanishing` | 1e-6 | **6e-2** (Bayes) | Sensibilité vanishing |
| `threshold_exploding` | 1e3 | **2e-1** (Bayes) | Sensibilité explosion |
| confidence boost coupling | 0.2 | ? | Poids des couplages |
| saturation_threshold | 0.5 | ? | Seuil saturation actv. |

#### Algorithmes d'optimisation
| Algo | Principe | Coût | Quand l'utiliser |
|------|----------|------|-----------------|
| **Grid Search** | Tester TOUTES les combinaisons d'une grille fixe | Élevé (N^params) | ≤ 2 params, exploratoire |
| **Random Search** | Tester N combinaisons aléatoires | Moyen | > 2 params, baseline |
| **Bayesian (GP)** | Modéliser l'accuracy comme une fonction gaussienne, explorer les zones prometteuses | Faible (10-30× moins de runs) | Par défaut — le meilleur rapport coût/précision |
| **CMA-ES** | Évolution différentielle sans gradient | Moyen | Paramètres continus, surface non-lisse |

#### External Benchmarks
Ces benchmarks vérifient que l'amélioration sur notre dataset n'est pas du surapprentissage.

| Source | Type | Comment |
|--------|------|---------|
| **Issues GitHub PyTorch** | Réel | Scraper des issues avec stack traces, vérifier le diagnostic |
| **Papers de recherche** | Synthétique | Reproduire les modes de défaillance documentés |
| **Kaggle competitions** | Réel | Entraînements qui échouent, causes documentées |
| **HuggingFace Forums** | Réel | Posts "training failed" avec logs |
| **Student's mistakes book** | Pédagogique | Catalogue d'erreurs classiques en DL |
| **Auto-validation** | Automatique | Générer N runs aléatoires avec bugs aléatoires, mesurer l'accuracy |

### Phase 9 : Auto-Validation Continue
- [ ] Boucle CI : à chaque commit → run benchmark → vérifier accuracy ≥ seuil
- [ ] Auto-tune : si accuracy baisse → lancer Bayesian search → commit les nouveaux seuils
- [ ] Rapport hebdomadaire : évolution de l'accuracy dans le temps
- [ ] Détection de régression : alerter si l'accuracy baisse de > 0.05

### Phase 10 : MVP Launch & Market Validation (Go-to-Market)
> **Objectif** : Valider le Product-Market Fit avec des utilisateurs réels. Passer du "Code qui marche" au "Produit qu'on utilise".

#### 1. Métriques de Succès (KPIs de Validation)
| Métrique | Objectif (30 jours) | Pourquoi |
|---|---|---|
| **Installs PyPI** | 100 | Preuve que le packaging et l'install fonctionnent |
| **GitHub Stars** | 50 | Preuve d'intérêt et de crédibilité technique |
| **Retours Qualifiés** | 5 | Preuve que le problème est réel et que NeuralDBG aide |
| **Issues/PRs** | 3+ | Preuve que la communauté s'engage |

#### 2. Cibler les Early Adopters (Personas Phase 4)
- [ ] **Le PhD Student** (Reddit r/MachineLearning, r/deeplearning) : "J'ai passé 3 jours à debugger mon gradient, j'aurais aimé avoir ça."
- [ ] **Le ML Engineer** (HackerNews, Twitter/X) : "On a un pipeline CI/CD, mais on n'a pas de test de validité causale sur nos models."
- [ ] **Le Researcher** (Twitter/X, Discord IA) : "Je publie un paper, j'ai besoin d'expliquer *pourquoi* mon model a divergé."

#### 3. Préparer les Assets de Lancement
- [x] **README "Killer"** : Section "Why NeuralDBG?" avec comparatif visuel.
- [x] **Landing Page** (GitHub Pages) : Page statique dans `docs/index.html`.
- [x] **Démo Vidéo** : Script `scripts/record_demo.py` pour enregistrement automatisé.
- [x] **Exemple "Copy-Paste"** : `quickstart.py` à la racine.

#### 4. Canaux de Distribution (Launch Plan)
- [ ] **Reddit** : Poster sur r/MachineLearning.
- [x] **HackerNews** : Brouillon prêt dans `LAUNCH_POSTS.md`.
- [ ] **Twitter/X** : Thread avec le problème/solution.
- [ ] **Discord** : Partager dans les serveurs IA/ML.

#### 5. Boucle de Feedback (Validation Loop)
- [x] **GitHub Issues Template** : Bug, Feature, False Positive.
- [ ] **Discord Channel** : Créer un channel `#feedback`.
- [ ] **Mom Test 2.0** : Contacter les 5 premiers utilisateurs.
- [ ] **Discord Channel** : Créer un channel `#feedback` pour discuter en direct avec les premiers utilisateurs.
- [ ] **Mom Test 2.0** : Contacter les 5 premiers utilisateurs pour un call de 15 min. "Est-ce que ça t'a vraiment fait gagner du temps ?"

---
**Last Updated**: 2026-05-17
