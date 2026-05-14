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
| 🟡 Moyenne | **Distributed/DataParallel** | Multi-GPU hook integrity | ❌ |
| 🟢 Basse | **LSTM/Time Series** | Vanishing recurrent gradients | ❌ |
| 🟢 Basse | **GNN** (GCN/GAT) | Oversmoothing, deep GNN | ❌ |
| 🟢 Basse | **RL** (PPO/DQN) | Reward hacking, policy collapse | ❌ |
| 🟢 Basse | **torch.compile** | Dynamo graph compatibility | ❌ |

### Phase 3 : Pipeline Aquarium
- [ ] Ajouter `export_aquarium_package()` dans les 3 scénarios ResNet
- [ ] Générer un fichier `events.json` pour chaque scénario
- [ ] Vérifier le chargement dans Aquarium (`~/Documents/Aquarium/`)
- [ ] Itérer sur le format si nécessaire

### Phase 4 : Desk Research (R75) — MANDATORY ✅
- [x] Personas (4 personas avec verbatim Reddit/HN/SO)
- [x] Competitors (8 : W&B, Neptune, MLflow, TensorBoard, Captum, Comet, WhyLabs, OpenAI Clarity)
- [x] Market Sizing (TAM $16B, SAM $1.2B, SOM $2M/3yr)
- [x] Risk Analysis (5 risques avec probabilité/impact/remède)
- [x] Gap Analysis (4 gaps avec preuves)
- ✅ **GO Decision** — toutes les 5 dimensions complétées

### Phase 5 : Publication PyPI
- [ ] Ajouter `~/.pypirc` (token PyPI)
- [ ] Créer `.github/workflows/publish.yml`
- [ ] Builder + push sur PyPI
- [ ] Vérifier `pip install neuraldbg` fonctionnel

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

##### Restructuration du Code
- [ ] Séparer le moteur causal en `neuraldbg-engine/` (privé, nouveau repo)
- [ ] Laisser `neuraldbg-core/` (MIT, public) avec hooks + collecte d'events uniquement
- [ ] Le public `NeuralDbg` context manager appelle l'engine privé via import conditionnel
- [ ] Les tests de validité causale (Phase 0) restent privés

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

---
**Last Updated**: 2026-05-14
