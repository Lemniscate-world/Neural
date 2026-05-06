# Cours NeuralDBG — Pour mon associé

## 1. Le Problème

Quand tu entraines un réseau de neurones et que ça échoue, tu as deux outils principaux :
- **TensorBoard / W&B** — *"Quelle métrique est bonne ?"*
- **tensor inspection** — *"Quelle valeur est bizarre ?"*

**Mais personne ne répond à :** *"Pourquoi mon modèle a échoué ?"*

C'est exactement ce que NeuralDBG répond.

---

## 2. Qu'est-ce que NeuralDBG ?

**Un moteur d'inférence causale pour le debugging de réseaux de neurones.**

Au lieu de regarder des valeurs brutes (tensor inspection), NeuralDBG :
1. Capture des **événements sémantiques** (transitions, pas des valeurs)
2. Analyse les **patterns de propagation** entre couches
3. Génère des **hypothèses causales classées** par confiance

```
Training Loop → Semantic Events → Causal Analysis → Ranked Hypotheses
```

---

## 3. Les 4 Types d'Événements

| Type | Source | Détecte |
|------|--------|---------|
| `gradient_health_transition` | Hooks backward | Vanishing, exploding, saturation des gradients |
| `activation_regime_shift` | Hooks forward | Neurones morts, activations saturées |
| `optimizer_instability` | `record_loss()` | Plateaux, pics, divergence de la loss |
| `data_anomaly` | Hooks forward (inputs) | NaN, Inf, distribution shift |

**Pourquoi des événements et pas des tensors ?**
→ Compression 10,000x — on stocke les transitions, pas les valeurs

---

## 4. Comment ça marche (technique)

### Step 1: Extraction via Hooks
```python
with NeuralDbg(model) as dbg:
    for step, (inputs, targets) in enumerate(dataloader):
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        dbg.record_loss(loss.item())  # Pour optimizer instability
        optimizer.step()
```

Les hooks capturent des stats (mean, std, sparsity, saturation_ratio) — **pas les tensors**.

### Step 2: Compression en événements
```python
# Avant: 10,000 steps de données brutes
# Après: 15 événements sémantiques
```

### Step 3: Raisonnement causal (4 couches)
1. **First-Occurrence** — Quelle couche a échoué en premier ?
2. **Temporal Coupling** — Événements séquentiels dans une fenêtre de 5 steps
3. **Cross-Domain** — Corrélation entre gradients + activations + loss
4. **Pattern Matching** — Templates pré-définis ("vanishing via saturation")

### Step 4: Hypothèses classées
```python
explanations = dbg.explain_failure("vanishing_gradients")
# Output: "Gradient vanishing originated in layer 'linear1' at step 234, 
#          likely due to LR × activation mismatch (confidence: 0.87)"
```

---

## 5. Les 6 Types d'Échecs Supportés

| Échec | Méthode | Explication |
|-------|---------|-------------|
| `vanishing_gradients` | `_explain_vanishing_gradients()` | Cause racine + coupling saturation |
| `exploding_gradients` | `_explain_exploding_gradients()` | Première couche à exploser |
| `dead_neurons` | `_explain_dead_neurons()` | Mort de neurones dans les couches d'activation |
| `saturated_activations` | `_explain_saturated_activations()` | Patterns de saturation |
| `optimizer_instability` | `_explain_optimizer_instability()` | Plateaux, pics, divergence + cross-ref gradients |
| `data_anomaly` | `_explain_data_anomaly()` | NaN/Inf/distribution shift dans les inputs |

---

## 6. Démos Disponibles

### Demo 1: Vanishing Gradients
```bash
python demo_vanishing_gradients.py
```
Montre :
- Gradient health transition detecté
- Cause racine identifiée (couche + step)
- Confidence score

### Demo 2: Data Anomaly
```bash
python demo_data_anomaly.py
```
4 scénarios :
- NaN detection
- Distribution shift
- Optimizer instability
- Cross-domain coupling

### Dogfooding
```bash
python dogfooding_resnet.py
```
Testé sur ResNet-18 (11M params) — 561 événements capturés sur 30 steps

---

## 7. Architecture Technique

```
┌─────────────────────────────────────────────────┐
│                  NeuralDbg                       │
├─────────────────────────────────────────────────┤
│ 1. SemanticEventExtractor (hooks)               │
│    - Forward hooks → activations, data inputs    │
│    - Backward hooks → gradients                 │
│    - record_loss() → optimizer                   │
├─────────────────────────────────────────────────┤
│ 2. CausalCompressor (_collapse_events)          │
│    - Merge événements séquentiels               │
│    - Trace les transitions                      │
├─────────────────────────────────────────────────┤
│ 3. PostMortemReasoner (explain_failure)          │
│    - First-occurrence tracking                   │
│    - Temporal coupling (window=5)               │
│    - Cross-domain correlation                   │
│    - Pattern matching                           │
├─────────────────────────────────────────────────┤
│ 4. CausalGraphExporter (export_mermaid)         │
│    - Visualisation du graphe causal              │
└─────────────────────────────────────────────────┘
```

---

## 8. Compiler-Awareness (torch.compile)

```python
# Recommandé : wrap AVANT compilation
with NeuralDbg(model) as dbg:
    model_compiled = torch.compile(model)
    # ... training
```

Les hooks sont wrapés avec `@dynamo_disable` pour survivre à `torch.compile`.

---

## 9. Cas d'Usage Idéaux

**NeuralDBG est fait pour :**
- 🔬 ML Researchers cherchant des explications causales
- 🎓 PhD Students analysant des architecturesnovel
- ⚙️ Research Engineers comprennant l'instabilité d'optimization

**NeuralDBG n'est pas pour :**
- ❌ Production monitoring (utilise W&B/Prometheus)
- ❌ No-code users
- ❌ TensorFlow/JAX (PyTorch only)

---

## 10. Métriques du Projet

| Composant | Status |
|-----------|--------|
| 4 types d'événements | ✅ |
| 6 types d'échecs expliqués | ✅ |
| 79 tests | ✅ |
| 85% coverage | ✅ |
| bandit: 0 issues | ✅ |
| torch.compile compatible | ✅ |
| 2 démos fonctionnelles | ✅ |
| Dogfooding ResNet-18 | ✅ |

**Progress actuel** : 72% (jalon 5 en cours)

---

## 11. Commandes Utiles

```bash
# Setup
make bootstrap
source .venv/bin/activate

# Tests
pytest -v

# Demos
python demo_vanishing_gradients.py
python demo_data_anomaly.py

# Couverture
pytest --cov=neuraldbg --cov-report=term-missing
```

---

## 12. Pour Aller Plus Loin

Post-MVP (Phase 2) :
- Intégration Granger causality / Bayesian graphs
- Visualisation des explications (pas des tensors)
- Expansion des types de questions causales
- Formalisation de la sémantique d'inférence

---

## Résumé en 3 phrases

> NeuralDBG répond à "pourquoi mon modèle a échoué ?" — pas "quelle métrique est bonne ?"
> Il capture des événements sémantiques (transitions) et applique un raisonnement causal en 4 couches pour générer des hypothèses classées par confiance.
> C'est un outil de debugging pour researchers, pas un outil de monitoring pour la prod.