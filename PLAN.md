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

### Phase 4 : Desk Research (R75) — MANDATORY
- [ ] Personas (3-4 avec verbatim Reddit/HN)
- [ ] Competitors (5+ : WhyLabs, Weights & Biases, Neptune, TensorBoard, MLflow)
- [ ] Market Sizing (TAM/SAM/SOM)
- [ ] Risk Analysis (5 risques)
- [ ] Gap Analysis (3+ gaps)
- ⚠️ Règle 75 : Shallow research < 3 dimensions = STOP. Pas de GO sans ça.

### Phase 5 : Publication PyPI
- [ ] Ajouter `~/.pypirc` (token PyPI)
- [ ] Créer `.github/workflows/publish.yml`
- [ ] Builder + push sur PyPI
- [ ] Vérifier `pip install neuraldbg` fonctionnel

### Phase 6 : Agent Auto-Correcteur
- [ ] Créer repo `~/Documents/Neural-Agent/`
- [ ] Définir le protocole : NeuralDBG causal output → action
- [ ] Implémenter un agent qui reçoit `explain_failure()` → ajuste LR/init/archi
- [ ] Boucle fermée : training → diagnostic → correction → nouveau training

---
**Last Updated**: 2026-05-14
