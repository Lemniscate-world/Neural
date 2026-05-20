# HackerNews "Show HN" Draft — NeuralDBG

> ⚡ Mise à jour : 2026-05-18 — version finale prête pour lancement

---

## 1. Post Show HN

### Titre (max 80 caractères)
```
Show HN: NeuralDBG – Causal root cause analysis for PyTorch training
```

(76/80 caractères ✅)

### URL
```
https://github.com/LambdaSection/NeuralDBG
```

### Premier commentaire (auto-posté)

```markdown
I built NeuralDBG because I was tired of staring at TensorBoard curves guessing why my model failed.

Most debugging tools show you *when* the loss spiked or vanished, but they don't tell you *why*. NeuralDBG analyzes gradients, activations, and data during training to provide structured causal hypotheses:

"Gradient vanishing originated in layer 'linear1' at step 234, likely due to LR × activation mismatch (confidence: 0.87)"

It's a Python package you wrap around your training loop. No dashboard setup, no cloud account, 100% local.

Key features:
- Semantic event extraction (detects transitions like Healthy → Vanishing)
- Post-mortem reasoning with ranked hypotheses
- Optimizer instability detection (plateaus, spikes, divergence)
- Data anomaly detection (NaN, Inf, distribution shifts)
- Works with torch.compile and distributed training

MIT License, pip install neuraldbg. Feedback welcome!
```

---

## 2. X / Twitter Thread (R94)

### Post 1 (annonce)
```
NeuralDBG: causal root cause analysis for PyTorch training is now open source (MIT).

Why TensorBoard shows you WHEN your model fails. NeuralDBG tells you WHY.

🔗 github.com/LambdaSection/NeuralDBG

#NeuralDBG #PyTorch #ML
```

### Post 2 (le problème)
```
The problem: debugging ML training is trial-and-error. You see NaN loss, you guess: LR? Data? Architecture? Layer X? 

You waste days. 

We built a tool that answers: "layer 'linear1' at step 234, LR × ReLU mismatch (confidence: 0.87)".
```

### Post 3 (le différentiateur)
```
Existing tools show histograms (TensorBoard) or curves (W&B). None tell you WHY.

NeuralDBG detects causal chains: vanishing → which layer → why → confidence score.

MIT, local, works with torch.compile.
```

### Post 4 (call to action)
```
pip install neuraldbg

Star on GitHub if useful → github.com/LambdaSection/NeuralDBG

Feedback welcome. Built this because I needed it myself.

#NeuralDBG #ML #OpenSource
```

---

## 3. Reddit Post (r/MachineLearning)

### Titre
```
[Project] NeuralDBG – Causal root cause analysis for PyTorch training (open source)
```

### Contenu (markdown)
```markdown
## Le problème
Quand un training échoue (NaN loss, vanishing gradients), les outils existants (TensorBoard, W&B) montrent *quand* ça arrive mais pas *pourquoi*.

On passe des heures à debugger manuellement des bugs qui sont en fait des patterns silencieux et récurrents.

## Ce qu'on a construit
NeuralDBG analyse les activations, gradients et données pendant le training et répond :
> "Gradient vanishing originated in layer 'linear1' at step 234, likely due to LR × activation mismatch (confidence: 0.87)"

## Différence clé
- **TensorBoard** : histogrammes de gradients (tu regardes, tu devines)
- **W&B** : courbes de loss (tu regardes, tu devines)
- **NeuralDBG** : chaîne causale structurée avec module responsable + confiance

## Key features
- Semantic event extraction (Healthy → Vanishing → NaN)
- Post-mortem reasoning with ranked hypotheses
- Optimizer instability detection
- Data anomaly detection (NaN, Inf, distribution shifts)
- Works with torch.compile and distributed training

## Lien
https://github.com/LambdaSection/NeuralDBG

MIT, pip install neuraldbg, 100% local.

Des questions ? Des retours ? Je suis preneur.
```

---

## 4. Discord Posts

### FrancophonIA (#ia-general)
```
Salut ! J'ai bossé sur un outil de debug pour PyTorch qui analyse automatiquement les gradients et activations pour trouver la cause racine des NaN, vanishing, exploding, etc.

Concrètement au lieu de regarder des courbes TensorBoard en devinant d'où vient le problème, il te dit direct : "c'est le layer X à l'étape Y, probablement dû à Z".

C'est open source MIT, installable en pip install neuraldbg.

Des retours ? Des gens qui galèrent avec ce genre de problèmes ?
```

### PyTorch Discord (#showcase)
```
Hey! Been working on a debugging tool for PyTorch that does causal root cause analysis on training runs.

Instead of staring at TensorBoard curves guessing, it tells you: "gradient vanished in layer 'linear1' at step 234, likely due to LR × activation mismatch".

MIT, pip install neuraldbg, works with torch.compile and distributed.

Would love any feedback!
```

### Hugging Face Discord (#showcase)
```
Just open-sourced NeuralDBG – a tool that does causal root cause analysis for PyTorch training.

It detects semantic events (Healthy → Vanishing → NaN), generates ranked hypotheses with confidence scores.

Key differentiator: existing tools show WHEN. NeuralDBG answers WHY.

https://github.com/LambdaSection/NeuralDBG

MIT license. Feedback welcome!
```

---

## 5. Réponses types aux questions HN probables

### Q : "How is this different from TensorBoard?"
**R** : TensorBoard shows histograms of gradients — you still have to manually trace which layer caused the issue. NeuralDBG does that trace automatically and outputs a structured causal chain: layer → step → cause → confidence.

### Q : "Does it work with [framework]?"
**R** : Currently PyTorch native. Works with torch.compile, DDP, and FSDP. We're looking at JAX support — if that matters to you, open an issue or PR.

### Q : "Is this just a wrapper around backward hooks?"
**R** : Partially! We use full backward hooks (the non-deprecated kind since PyTorch 2.x). But the value is in the semantic analysis layer on top: event classification, coupled failure detection, and the explanation engine. The hooks are just the data source.

### Q : "What's the performance overhead?"
**R** : ~5-15% depending on model size and hook granularity. You can enable/disable per-module hooks. For debugging runs, it's negligible. For production training, you can run it on a subset of layers.

### Q : "Can it auto-fix the training?"
**R** : Not yet — but that's the direction. Currently it's diagnostic. We export structured data (JSON + Mermaid graphs) so AI agents can consume it. The auto-correction loop is on the roadmap.

### Q : "Why not just use W&B?"
**R** : W&B is great for experiment tracking. It shows you WHEN something happened. NeuralDBG is complementary — it tells you WHY. We actually export data that you can push to W&B.

### Q : "Is there a hosted version?"
**R** : No, 100% local. MIT open source. No cloud, no accounts, no data leaving your machine. If there's demand for a hosted version, we'll consider it.

---

## Fichiers associés
- `docs/launch_plan_neuraldbg.md` — planning temporel
- `docs/hn_feedback_log.md` — log des retours après lancement
- `docs/tracking/acquisition_tracker.md` — historique des posts