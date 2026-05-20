# RULE 96: Community Post Protocol (Reddit + Discord)

## Trigger
Lors d'un lancement (Show HN, release, ou milestone public).

## Principes généraux
- **Ne JAMAIS** poster un lien froid sans contexte
- **Toujours** cadrer le post comme une question ou un problème, pas une promo
- **Toujours** répondre aux commentaires dans les 24h
- **Jamais** de cross-post identique — adapter le ton à chaque plateforme

---

## Reddit

### Subreddits cibles
| Subreddit | Type de post | Tag |
|---|---|---|
| `r/MachineLearning` | Technique, problème/solution | [Discussion] ou [Project] |
| `r/PyTorch` | Technique détaillée | [P] |
| `r/rust` (si Rust utilisé) | Technique infra | — |
| `r/opensource` | Philosophy, contribution | — |

### Template de post Reddit

**Titre** :
```
[Project] NeuralDBG – Causal root cause analysis for PyTorch training (open source)
```

**Contenu** (markdown) :
```
## Le problème
Quand un training échoue (NaN loss, vanishing gradients), les outils existants (TensorBoard, W&B) montrent *quand* ça arrive mais pas *pourquoi*.

On passe des heures à debugger manuellement des bugs qui sont en fait des patterns silencieux et récurrents.

## Ce qu'on a construit
NeuralDBG analyse les activations, gradients et données pendant le training et répond :
> "La perte a explosé à l'étape 234, origine dans 'layer4.0.conv1' — cause probable : learning rate trop élevé combiné à une activation ReLU en saturation."

## Différence clé
- **TensorBoard** : histogrammes de gradients (tu regardes, tu devines)
- **W&B** : courbes de loss (tu regardes, tu devines)
- **NeuralDBG** : chaîne causale structurée avec module responsable + confiance

## Lien
https://github.com/LambdaSection/NeuralDBG

MIT, pip install neuraldbg, 100% local.

Des questions ? Des avis ? Je suis preneur.
```

### Règles Reddit
- Ne pas poster plus d'1 fois par mois sur le même subreddit
- Ne pas DM les modérateurs
- Si le post est supprimé : ne pas insister, passer au subreddit suivant
- Loguer le résultat dans `docs/tracking/acquisition_tracker.md`

---

## Discord

### Serveurs cibles
| Serveur | Salon | Ton |
|---|---|---|
| **FrancophonIA** | #ia-general ou #projets | Technique, décontracté |
| **Serveur Neural (si existe)** | #showcase ou #dev | Technique |
| **PyTorch Discord** | #showcase | Anglais, technique |
| **Hugging Face Discord** | #showcase | Anglais, orienté ML |

### Template de post Discord

**FrancophonIA :**
```
Salut ! J'ai bossé sur un outil de debug pour PyTorch qui analyse automatiquement les gradients et activations pour trouver la cause racine des NaN, vanishing, exploding, etc.

Concrètement au lieu de regarder des courbes TensorBoard en devinant d'où vient le problème, il te dit direct : "c'est le layer X à l'étape Y, probablement dû à Z".

C'est open source MIT, installable en `pip install neuraldbg`. Si ça intéresse du monde je peux faire une démo rapide.

Des retours ? Des gens qui galèrent avec ce genre de problèmes ?
```

**Anglais (PyTorch / HF Discord) :**
```
Hey! Been working on a debugging tool for PyTorch that does causal root cause analysis on training runs.

Instead of staring at TensorBoard curves guessing, it tells you: "gradient vanished in layer 'linear1' at step 234, likely due to LR × activation mismatch".

MIT, pip install neuraldbg, works with torch.compile and distributed.

Would love any feedback!
```

### Règles Discord
- Pas de lien dans le premier message (sauf si le salon l'autorise)
- Toujours répondre aux questions avant de reposter ailleurs
- Ne pas spammer les DMs des membres

---

## Logging

Après chaque post, ajouter une entrée dans `docs/tracking/acquisition_tracker.md` :

```markdown
| Plateforme | Audience | Post | Résultat |
|---|---|---|---|
| Reddit r/MachineLearning | Devs ML | Lien vers post | X upvotes, Y commentaires |
| Discord FrancophonIA | Devs FR | Lien vers salon | Z réponses, feedback : ... |