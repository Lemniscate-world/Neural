# Community Post Template — NeuralDBG

> Généré par R77 — L2 Auto-Distribution Pipeline
> Date : 2026-05-18

---

## Reddit — r/MachineLearning

**Titre :**
```
[Project] NeuralDBG – Causal root cause analysis for PyTorch training (open source)
```

**Contenu :**
```markdown
## Le problème
Quand un training échoue (NaN loss, vanishing), les outils existants montrent *quand* ça arrive mais pas *pourquoi*.

## Ce qu'on a construit
NeuralDBG analyse les activations, gradients et données pendant le training et répond :
"Gradient vanishing originated in layer 'linear1' at step 234, due to LR × activation mismatch (conf: 0.87)"

## Différence clé
- TensorBoard : histogrammes (tu regardes, tu devines)
- NeuralDBG : chaîne causale structurée

MIT, pip install neuraldbg

https://github.com/LambdaSection/NeuralDBG
```

---

## Discord

**FrancophonIA :**
```
Salut ! J'ai bossé sur un outil de debug pour PyTorch qui analyse automatiquement les gradients et activations pour trouver la cause racine des NaN, vanishing, etc. C'est open source MIT, installable en pip install neuraldbg. Des retours ?
```

**PyTorch Discord :**
```
Hey! Built NeuralDBG – causal root cause analysis for PyTorch training. It tells you WHY your model failed, not just WHEN. MIT, pip install neuraldbg. Feedback welcome!
```

---

## X / Twitter

```
NeuralDBG: causal root cause analysis for PyTorch training is now open source (MIT).

Why TensorBoard shows you WHEN your model fails. NeuralDBG tells you WHY.

🔗 github.com/LambdaSection/NeuralDBG

#NeuralDBG #PyTorch
```

---

## Logging

Après chaque post, mettre à jour `docs/tracking/acquisition_tracker.md` et `docs/hn_feedback_log.md`.