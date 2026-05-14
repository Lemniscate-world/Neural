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

### Phase 0 : Validité Causale — Moteur Infaillible
Le vrai moat n'est pas le code, c'est la **qualité du raisonnement causal**.
Ces tests prouvent que le moteur ne peut pas être cloné sans être réécrit au même niveau de rigueur.

| Priorité | Test | Pourquoi | Statut |
|----------|------|----------|--------|
| 🔴 **Critique** | **Validité causale** : injecter NaN dans une couche spécifique → vérifier que l'engine localise *cette couche* et pas une autre | Seul test qui distingue causalité VS corrélation | ❌ |
| 🔴 **Critique** | **Faux positifs** : entraînement sain (LR optimal, init correcte) → 0 hypothèses, 0 alertes | Sans ça, l'engine crie au loup en production | ❌ |
| 🔴 **Critique** | **Déterminisme** : même seed + même bug → même diagnostic exact (hash des hypothèses) | Reproductibilité pour debug CI | ❌ |
| 🟡 **Haut** | **Mutation** : casser un modèle de N façons connues (N modes de défaillance) → engine détecte les N | Couverture exhaustive, pas de trou | ❌ |
| 🟡 **Haut** | **Scalabilité** : modèle avec 1000 modules feuilles → hooks s'installent en < 1s | Preuve que ça passe à l'échelle | ❌ |
| 🟡 **Haut** | **API Contract** : `export_aquarium_package()` → JSON valide conforme à un schéma connu | Interopérabilité avec Aquarium & Agent | ❌ |
| 🟢 **Moyen** | **Invariance cross-architecture** : NaN dans MLP = même diagnostic que NaN dans ResNet = même diagnostic que NaN dans Transformer | Cohérence du diagnostic | ❌ |
| 🟢 **Moyen** | **Régression CI** : hook dans la CI qui bloque si le nombre d'hypothèses change inexpliqué | Détection précoce de dérive | ❌ |

### Phase 2 : Dogfooding Extensif
| Priorité | Architecture | Type de défaillance | Statut |
|----------|-------------|---------------------|--------|
| 🔴 Haute | **Transformer** (nanoGPT) | Attention collapse, NaN softmax, LR warmup | ❌ |
| 🔴 Haute | **GANs** (DCGAN) | Mode collapse, D/G imbalance | ❌ |
| 🟡 Moyenne | **LLM fine-tuning** (LoRA) | Catastrophic forgetting, loss spikes | ❌ |
| 🟡 Moyenne | **Diffusion** (DDPM) | Unstable denoising, NaN UNet | ❌ |
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
- [ ✓ ] Créer repo `~/Documents/Neural-Agent/`
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

---
**Last Updated**: 2026-05-14
