# Phase 2 — Dogfooding Extensif : Documentation Complète

## Glossaire des Concepts

### Poids Récurrents (Recurrent Weights)

Dans un réseau récurrent (RNN, LSTM, GRU), il y a **deux types de poids** :

```
h(t) = σ(W_input · x(t)  +  W_recurrent · h(t-1)  +  b)
        ↑                    ↑
        poids d'entrée       poids récurrents
```

- **Poids d'entrée (`W_input`)** : transforment l'input courant `x(t)`
- **Poids récurrents (`W_recurrent`)** : transforment l'état caché précédent `h(t-1)`

Les poids récurrents sont le **mécanisme de mémoire** du réseau. Ils déterminent combien d'information du passé est conservée ou oubliée. Si on les met à zéro → le réseau n'a plus de mémoire, chaque step est indépendant → les gradients ne circulent plus dans le temps → **vanishing gradients temporels**.

---

### GCN (Graph Convolutional Network)

Un **GCN** est un réseau de neurones qui opère sur des **graphes** (pas sur des images ou du texte).

```
Noeuds : entités (utilisateurs, molécules, documents...)
Arêtes : relations (amis, liaisons chimiques, citations...)
```

Contrairement à un MLP classique qui fait `output = σ(W · x + b)`, un GCN fait :

```
H' = σ(Ã · H · W)
      ↑    ↑    ↑
      adj  feat poids
```

- **`Ã` (matrice d'adjacence normalisée)** : propage l'information entre noeuds connectés
- **`H` (features des noeuds)** : représentation de chaque noeud
- **`W` (poids appris)** : transformation linéaire

**Problème classique : l'oversmoothing**
Quand on empile trop de couches GCN (≥ 4-5), tous les noeuds finissent avec la **même représentation** (mélange excessif via la matrice d'adjacence). C'est l'équivalent du vanishing gradient pour les graphes.

---

### Dynamo (torch.compile)

**Dynamo** est le compilateur interne de PyTorch, activé via `torch.compile()`.

```python
model = torch.compile(model)  # ← Dynamo prend le relais
```

**Ce qu'il fait :**
1. Capture le graphe de calcul (forward pass)
2. L'optimise (fusion d'ops, élimination de code mort, kernel fusion)
3. Génère un binaire optimisé (via Triton ou CPU backend)

**Résultat :** 2-4× plus rapide en général.

**Problème pour NeuralDBG :** `torch.compile` réécrit le graphe de calcul. Les **hooks PyTorch** (forward_hook, backward_hook) que NeuralDBG installe peuvent être **ignorés ou modifiés** par le compilateur. Il faut vérifier que nos hooks survivent à la compilation.

---

## Scénarios Détaillés par Architecture

---

### 1. LSTM / Time Series

**Fichier :** `examples/demo_lstm_failures.py`
**Tests :** `tests/integration/test_lstm_demo.py` (5 tests)

#### Scénario A : Vanishing Recurrent

| Élément | Détail |
|---|---|
| **Bug injecté** | `weight_hh` (poids récurrents) mis à `0.0` sur toutes les couches |
| **Architecture** | LSTMForecaster : 3 couches LSTM, hidden_size=32, input_size=4 |
| **Pourquoi ça fail** | Avec `W_recurrent = 0`, la formule devient `h(t) = σ(W_input · x(t) + b)`. Le terme `h(t-1)` disparaît. Le réseau n'a **aucune mémoire temporelle**. Les gradients ne se propagent pas dans le temps → vanishing. |
| **LR utilisée** | 1e-3 |
| **Ce qu'on vérifie** | `len(events) > 0` — NeuralDBG capture au moins un événement (activation shift ou gradient vanishing) |

#### Scénario B : Exploding Recurrent

| Élément | Détail |
|---|---|
| **Bug injecté** | `weight_hh` multiplié par `50.0` sur toutes les couches |
| **Architecture** | Même LSTMForecaster 3 couches |
| **Pourquoi ça fail** | Avec `W_recurrent × 50`, chaque step amplifie le signal précédent de ×50. Après 3 couches et quelques steps, les activations deviennent énormes → gradients explosifs → `NaN` ou overflow. |
| **LR utilisée** | 1e-2 (haute pour amplifier l'effet) |
| **Ce qu'on vérifie** | `len(events) > 0` — détection d'explosion ou d'instabilité |

#### Scénario C : Deep LSTM

| Élément | Détail |
|---|---|
| **Bug injecté** | Pas de bug artificiel — 6 couches LSTM empilées |
| **Architecture** | LSTMForecaster 6 couches, hidden_size=16 |
| **Pourquoi ça fail** | Même sans bug, 6 couches LSTM avec un LR faible (1e-4) créent un **vanishing gradient naturel**. Le gradient doit traverser 6 couches × le temps → il s'atténue exponentiellement. |
| **LR utilisée** | 1e-4 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection de vanishing dans les couches profondes |

#### Tests unitaires

| Test | Vérification |
|---|---|
| `test_vanishing_recurrent_captures_events` | Events capturés > 0 |
| `test_exploding_recurrent_captures_events` | Events capturés > 0 |
| `test_deep_lstm_captures_events` | Events capturés > 0 |
| `test_lstm_mermaid_graph` | Graph Mermaid commence par `graph TD` |
| `test_lstm_couplings_deduplicated` | Aucune paire (trigger, consequence) dupliquée |

---

### 2. GNN (GCN / GAT)

**Fichier :** `examples/demo_gnn_failures.py`
**Tests :** `tests/integration/test_gnn_demo.py` (4 tests)

#### Scénario A : Oversmoothing (Deep GCN)

| Élément | Détail |
|---|---|
| **Bug injecté** | 8 couches GCN **sans LayerNorm** |
| **Architecture** | GCN : 8 couches, in_features=16, hidden=32, out_classes=5 |
| **Pourquoi ça fail** | Chaque couche GCN propage l'information via la matrice d'adjacence `Ã`. Après 8 multiplications successives, tous les noeuds convergent vers la **même représentation** (vecteur propre dominant de `Ã`). Les gradients vanissent car toutes les sorties deviennent identiques → plus de signal discriminant. |
| **LR utilisée** | 1e-3 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection de vanishing ou saturation |

#### Scénario B : GNN Exploding

| Élément | Détail |
|---|---|
| **Bug injecté** | Tous les poids multipliés par `100.0` |
| **Architecture** | GCN 3 couches, mêmes dimensions |
| **Pourquoi ça fail** | Chaque couche amplifie le signal de ×100. Après 3 couches : ×100³ = ×1,000,000. Les activations explosent → gradients énormes → instabilité. |
| **LR utilisée** | 1e-1 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection d'explosion |

#### Scénario C : GNN NaN

| Élément | Détail |
|---|---|
| **Bug injecté** | Premier paramètre du modèle mis à `float("nan")` |
| **Architecture** | GCN 3 couches |
| **Pourquoi ça fail** | Un seul `NaN` dans les poids contamine toutes les opérations qui l'utilisent. Après un forward pass, les activations deviennent `NaN` → le loss devient `NaN` → les gradients deviennent `NaN`. |
| **LR utilisée** | 1e-3 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection d'anomalie (NaN ou instabilité) |

#### Tests unitaires

| Test | Vérification |
|---|---|
| `test_oversmoothing_captures_events` | Events capturés > 0 |
| `test_gnn_exploding_captures_events` | Events capturés > 0 |
| `test_gnn_nan_detected` | Events > 0 ET (hypothèses NaN OU events détectés) |
| `test_gnn_mermaid_graph` | Graph Mermaid valide |

---

### 3. torch.compile (Dynamo)

**Fichier :** `examples/demo_torch_compile.py`
**Tests :** `tests/integration/test_torch_compile_demo.py` (4 tests)

#### Scénario A : Compiled Healthy

| Élément | Détail |
|---|---|
| **Bug injecté** | Aucun — entraînement sain |
| **Architecture** | SimpleMLP : 3 couches, input=32, hidden=64 |
| **Compilation** | `torch.compile(model)` appliqué AVANT l'entraînement |
| **Pourquoi ce test** | Vérifier que les hooks NeuralDBG **fonctionnent toujours** après compilation. Dynamo réécrit le graphe — si nos hooks sont ignorés, on ne capture rien. |
| **LR utilisée** | 1e-3 |
| **Ce qu'on vérifie** | `len(events) >= 0` — au minimum pas de crash. Les events peuvent être 0 si l'entraînement est sain. |

#### Scénario B : Compiled + Vanishing

| Élément | Détail |
|---|---|
| **Bug injecté** | Tous les poids × `1e-8` + `torch.compile` |
| **Architecture** | SimpleMLP 3 couches |
| **Pourquoi ça fail** | Poids minuscules → activations quasi-nulles → gradients quasi-nuls. Le compilateur ne corrige pas ça — il optimise juste un calcul qui ne produit rien. |
| **LR utilisée** | 1e-6 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection de vanishing **malgré** la compilation |

#### Scénario C : Compiled + Exploding

| Élément | Détail |
|---|---|
| **Bug injecté** | Tous les poids × `1000.0` + `torch.compile` |
| **Architecture** | SimpleMLP 3 couches |
| **Pourquoi ça fail** | Poids énormes → activations énormes → gradients énormes. Dynamo compile un calcul qui explose — il ne le stabilise pas. |
| **LR utilisée** | 1e-1 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection d'explosion **malgré** la compilation |

#### Tests unitaires

| Test | Vérification |
|---|---|
| `test_compile_healthy_captures_events` | Pas de crash, events >= 0 |
| `test_compile_vanishing_captures_events` | Events capturés > 0 |
| `test_compile_exploding_captures_events` | Events capturés > 0 |
| `test_compile_mermaid_graph` | Graph Mermaid valide |

---

### 4. RL (PPO-style)

**Fichier :** `examples/demo_rl_failures.py`
**Tests :** `tests/integration/test_rl_demo.py` (5 tests)

#### Architecture Actor-Critic

```
État → [Policy Network] → Distribution d'actions (logits)
     → [Value Network]   → Estimation de la valeur V(s)
```

Le Policy Network décide **quoi faire**, le Value Network estime **combien ça vaut**.

#### Scénario A : Policy Collapse

| Élément | Détail |
|---|---|
| **Bug injecté** | Tous les poids de la Policy mis à `0.0` |
| **Architecture** | ActorCritic : Policy (3 couches, 64 hidden) + Value (3 couches, 64 hidden) |
| **Pourquoi ça fail** | Policy weights = 0 → tous les logits sont égaux → la distribution est **uniforme** → l'agent choisit au hasard. Le gradient de policy `∇log π(a|s) · A` est quasi-nul car π est plat → **vanishing gradient**. L'apprentissage est bloqué. |
| **LR utilisée** | 1e-5 (très faible, amplifie le vanishing) |
| **Ce qu'on vérifie** | `len(events) > 0` — détection de vanishing |

#### Scénario B : Value Explosion

| Élément | Détail |
|---|---|
| **Bug injecté** | Poids du Value Network × `1000.0` |
| **Architecture** | Même ActorCritic |
| **Pourquoi ça fail** | Value Network avec poids énormes → V(s) prédit des valeurs gigantesques → le loss `(V(s) - R)²` explose → le gradient du Value Network explose → il contamine le gradient total (policy_loss + 0.5 * value_loss). |
| **LR utilisée** | 1e-1 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection d'explosion |

#### Scénario C : Reward Hacking

| Élément | Détail |
|---|---|
| **Bug injecté** | Rewards multipliés par `1e6` |
| **Architecture** | Même ActorCritic, poids normaux |
| **Pourquoi ça fail** | Rewards × 1e6 → l'avantage `A = R - V(s)` est énorme → le gradient de policy `∇log π · A` est énorme → chaque update change la policy de façon brutale → **instabilité**. La policy oscille wildly entre actions. |
| **LR utilisée** | 3e-4 (normale pour PPO) |
| **Ce qu'on vérifie** | `len(events) > 0` — détection d'instabilité optimizer |

#### Tests unitaires

| Test | Vérification |
|---|---|
| `test_policy_collapse_captures_events` | Events capturés > 0 |
| `test_value_explosion_captures_events` | Events capturés > 0 |
| `test_reward_hacking_captures_events` | Events capturés > 0 |
| `test_rl_mermaid_graph` | Graph Mermaid valide |
| `test_rl_couplings_deduplicated` | Aucune paire dupliquée |

---

### 5. Distributed / DataParallel

**Fichier :** `examples/demo_distributed_failures.py`
**Tests :** `tests/integration/test_distributed_demo.py` (4 tests)

#### Qu'est-ce que DataParallel ?

```
GPU 0 : forward(x[0:batch/2]) → loss[0] → backward → grad[0]
GPU 1 : forward(x[batch/2:])  → loss[1] → backward → grad[1]
         ↓
    Reduction : grad = (grad[0] + grad[1]) / 2
         ↓
    Update des poids sur chaque GPU
```

DataParallel **duplique le modèle** sur chaque GPU, split le batch, et synchronise les gradients.

#### Scénario A : DataParallel Healthy

| Élément | Détail |
|---|---|
| **Bug injecté** | Aucun |
| **Architecture** | ParallelMLP : 3 couches, input=32, hidden=64 |
| **Wrapper** | `nn.DataParallel(model)` (si ≥ 2 GPUs, sinon modèle nu) |
| **Pourquoi ce test** | Vérifier que les hooks NeuralDBG fonctionnent sur un modèle **wrapped** par DataParallel. Le wrapper change la structure des modules (`model.module` vs `model`). |
| **LR utilisée** | 1e-3 |
| **Ce qu'on vérifie** | `len(events) >= 0` — pas de crash, hooks fonctionnent |

#### Scénario B : DataParallel + Vanishing

| Élément | Détail |
|---|---|
| **Bug injecté** | Poids × `1e-8` + DataParallel |
| **Architecture** | ParallelMLP 3 couches |
| **Pourquoi ça fail** | Même vanishing que le cas non-distribué, mais on vérifie que les hooks **capturent le problème** même quand le modèle est répliqué sur plusieurs GPUs. |
| **LR utilisée** | 1e-6 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection de vanishing en mode distribué |

#### Scénario C : DataParallel + Exploding

| Élément | Détail |
|---|---|
| **Bug injecté** | Poids × `1000.0` + DataParallel |
| **Architecture** | ParallelMLP 3 couches |
| **Pourquoi ça fail** | Même explosion que le cas non-distribué. On vérifie que les gradients explosifs sont détectés **avant** la réduction entre GPUs. |
| **LR utilisée** | 1e-1 |
| **Ce qu'on vérifie** | `len(events) > 0` — détection d'explosion en mode distribué |

#### Tests unitaires

| Test | Vérification |
|---|---|
| `test_dp_healthy_captures_events` | Pas de crash, events >= 0 |
| `test_dp_vanishing_captures_events` | Events capturés > 0 |
| `test_dp_exploding_captures_events` | Events capturés > 0 |
| `test_dp_mermaid_graph` | Graph Mermaid valide |

---

### 6-9. Architectures pré-existantes (déjà documentées)

| Architecture | Fichier | Scénarios | Tests |
|---|---|---|---|
| **Transformer** | `demo_transformer_failures.py` | No warmup, no norm, no scale | 5 |
| **GAN** | `demo_gan_failures.py` | Vanishing, exploding, NaN generator | 4 |
| **LoRA** | `demo_lora_finetune.py` | NaN, exploding, catastrophic forgetting | 3 |
| **Diffusion** | `demo_diffusion_failures.py` | NaN UNet, exploding, noise collapse | 4 |

---

## Résumé Global

### Ce qu'on a vérifié pour CHAQUE scénario

| Vérification | Méthode | Pourquoi |
|---|---|---|
| **Events capturés** | `len(results["events"]) > 0` | Les hooks forward/backward fonctionnent sur cette architecture |
| **Mermaid graph** | `results["mermaid"].startswith("graph TD")` | L'export du graphe causal ne crash pas |
| **Couplings dédupliqués** | Aucune paire `(trigger, consequence)` en double | Pas de faux doublons dans les couplages de défaillances |
| **Hypothèses causales** | `len(results["hypotheses"]) > 0` (quand applicable) | Le moteur génère des explications pertinentes |

### Bug injecté → Type de détection attendu

| Bug | Type d'événement attendu |
|---|---|
| Poids → 0 | `vanishing_gradients` |
| Poids × 1000 | `exploding_gradients` |
| NaN dans poids | `data_anomaly` |
| Architecture profonde | `vanishing_gradients` (naturel) |
| Rewards × 1e6 | `optimizer_instability` |
| Pas de LayerNorm | `activation_regime_shift` (saturation) |
| Pas de LR warmup | `exploding_gradients` (early training) |

### Fichiers créés/modifiés

| Fichier | Type | Description |
|---|---|---|
| `examples/demo_lstm_failures.py` | Nouveau | 3 scénarios LSTM |
| `examples/demo_gnn_failures.py` | Nouveau | 3 scénarios GCN/GAT |
| `examples/demo_torch_compile.py` | Nouveau | 3 scénarios torch.compile |
| `examples/demo_rl_failures.py` | Nouveau | 3 scénarios RL/PPO |
| `examples/demo_distributed_failures.py` | Nouveau | 3 scénarios DataParallel |
| `tests/integration/test_lstm_demo.py` | Nouveau | 5 tests LSTM |
| `tests/integration/test_gnn_demo.py` | Nouveau | 4 tests GNN |
| `tests/integration/test_torch_compile_demo.py` | Nouveau | 4 tests torch.compile |
| `tests/integration/test_rl_demo.py` | Nouveau | 5 tests RL |
| `tests/integration/test_distributed_demo.py` | Nouveau | 4 tests DataParallel |
| `neuraldbg/__init__.py` | Modifié | Fallbacks engine pour open-source |
| `PLAN.md` | Modifié | Phase 2 : 9/9 ✅ |
| `CHANGELOG.md` | Modifié | Changelog Unreleased |
