# RULE 95: Show HN Launch Protocol

## Trigger
Quand le produit est prêt (code fonctionnel, open source, documentation lisible) ET que la desk research (R75) est complète.

## Timing
- **1 unique lancement possible** sur HN — ne pas brûler
- Lancer un **mardi ou mercredi** (meilleurs jours pour HN), entre 10h et 12h ET (traffic US + Europe)
- Préparer 2h de disponibilité pour répondre aux commentaires

## Contenu du post

### Titre (max 80 caractères)
Format : `Show HN: NeuralDBG – [valeur unique en 5-8 mots]`

Exemple : `Show HN: NeuralDBG – Causal inference for PyTorch training dynamics`

### URL
Dépôt GitHub du projet (pas de landing page, pas de blog)

### Premier commentaire (auto-post)
Doit contenir :
1. **1 phrase** : ce que fait le projet
2. **Le problème** : pourquoi ça existe (pain point)
3. **La différence** : ce qui le rend unique vs TensorBoard, W&B, etc.
4. **1 lien** vers GitHub
5. **Pas de demande** ("star please", "feedback welcome" — sous-entendu)

## Contenu interdit
- Promesses sur des features futures ("soon will support...")
- Comparaisons agressives ("X is dead")
- Chiffres de croissance ou d'adoption
- Hype, exagérations, "revolutionary", "game-changing"

## Actions avant le lancement

### J-3 : Préparer les amplifications
- [ ] Rédiger le X post (R94) annonçant le HN
- [ ] Rédiger le post Reddit (r/MachineLearning)
- [ ] Rédiger le post Discord (FrancophonIA + serveur Neural)
- [ ] Préparer 3-5 réponses types aux questions probables

### J-1 : Vérifications techniques
- [ ] README.md lisible et à jour
- [ ] `pip install neuraldbg` fonctionne (test en venv frais)
- [ ] Quickstart fonctionne (1 copier-coller)
- [ ] GitHub Actions passent
- [ ] License visible (MIT)
- [ ] Lien vers la doc ou un notebook d'exemple

### J : Lancement
1. **Poster sur HN** → `https://news.ycombinator.com/submit`
2. **Attendre 5 min** → vérifier que le post est visible
3. **Auto-poster le premier commentaire** (texte préparé)
4. **Attendre 15 min** → poster le X thread
5. **Attendre 30 min** → poster sur Reddit
6. **Attendre 1h** → poster sur Discord

### J+1 : Suivi
- Répondre à TOUS les commentaires HN dans les 24h
- Loguer les retours dans `docs/hn_feedback_log.md`
- Mettre à jour le README si des questions reviennent

## Amplification différée (J+7)
- [ ] Écrire un blog technique (dev.to, medium) : "How we built..."
- [ ] Si le HN a bien marché (>100 upvotes) : préparer un post "Show HN: What we learned"
- [ ] Si le HN a peu marché (<20 upvotes) : documenter les hypothèses d'échec

## Métriques de succès
| Niveau | Upvotes | Commentaires | Signaux forts |
|--------|---------|-------------|---------------|
| 🟢 Excellent | >200 | >50 | Issues GitHub, PRs, questions techniques profondes |
| 🟡 Bien | 50-200 | 20-50 | Plusieurs étoiles GitHub, mentions Twitter |
| 🟠 Faible | 10-50 | 5-20 | Quelques étoiles, peu d'engagement |
| 🔴 Échec | <10 | <5 | Aucun signal — documenter pourquoi |

## Règle d'or
**HN n'est pas un canal de vente, c'est un canal de feedback technique.**
- Tu ne vends pas, tu montres
- Tu ne convaincs pas, tu expliques
- Tu ne défends pas, tu écoutes

## Fichiers associés
- `LAUNCH_POSTS.md` — drafts des posts (HN, X, Reddit, Discord)
- `docs/hn_feedback_log.md` — log des retours HN
- `docs/community_post_template.md` — templates génériques