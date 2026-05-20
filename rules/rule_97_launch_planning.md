# RULE 97: Launch Planning Master Template

## Trigger
À chaque fois qu'un projet est prêt pour distribution / lancement public.

## Structure du planning

Un planning de lancement est un document unique qui couvre **J-7 à J+30**.
Il est stocké dans `docs/launch_plan_{PROJECT_NAME}.md`.

---

## Phases temporelles

### T-7 : Préparation
- [ ] Finaliser le produit / la version
- [ ] Desk research terminée (R75)
- [ ] README.md à jour et lisible
- [ ] Quickstart fonctionnel (fresh venv)
- [ ] License visible

### T-3 : Création des assets
- [ ] Rédiger le post Show HN (R95)
- [ ] Rédiger le X thread (R94)
- [ ] Rédiger les posts communauté (R96)
- [ ] Préparer 3-5 réponses types aux questions HN probables
- [ ] Préparer une landing page si applicable (R56)

### T-1 : Vérifications finales
- [ ] Tests CI passent
- [ ] GitHub Actions vert
- [ ] `pip install` fonctionne
- [ ] Démo / quickstart fonctionne en 1 copier-coller
- [ ] Site web / landing page accessible

### J : Lancement
**Timing : mardi ou mercredi, 10h-12h ET**
```
08:00 ET → Vérification finale de tous les assets
10:00 ET → POSTER SHOW HN (news.ycombinator.com/submit)
10:05 ET → Vérifier que le post est visible
10:10 ET → Auto-poster le premier commentaire HN
10:30 ET → Post X thread (R94)
11:00 ET → Post Reddit (R96)
12:00 ET → Post Discord (R96)
```

### J+1 : Suivi
- [ ] Répondre à TOUS les commentaires HN
- [ ] Loguer les retours dans `docs/hn_feedback_log.md`
- [ ] Mettre à jour le README si des questions reviennent souvent

### J+7 : Amplification
- [ ] Blog technique (dev.to, medium)
- [ ] Si >100 upvotes HN : post "What we learned"
- [ ] Si faible engagement : analyser pourquoi (docs/launch_postmortem.md)

### J+30 : Bilan
- [ ] Compiler les métriques (stars GitHub, téléchargements PyPI, emails capturés)
- [ ] Documenter les leçons apprises
- [ ] Décider de la prochaine étape (v2, pivot, scaling)

---

## Canaux de distribution (par priorité)

| Priorité | Canal | Audience | Effort | ROI estimé |
|---|---|---|---|---|
| 1 | **Show HN** | ML engineers, fondateurs tech, investisseurs tech | 1h | ⭐⭐⭐⭐⭐ |
| 2 | **X / Twitter** | ML communauté technique | 30min | ⭐⭐⭐ |
| 3 | **Reddit r/MachineLearning** | Devs ML, chercheurs | 15min | ⭐⭐⭐ |
| 4 | **Discord** (FrancophonIA, PyTorch, HF) | Niche technique engagée | 15min | ⭐⭐ |
| 5 | **Blog technique** (dev.to) | SEO long terme, inbound | 2h | ⭐⭐ |
| 6 | **Newsletter ML** (TLDR AI, TheBatch) | Audience qualifiée | 1h | ⭐⭐ |
| 7 | **LinkedIn** (post technique) | Professionnels ML, recruteurs | 15min | ⭐ |
| 8 | **YouTube demo** | Devs qui préfèrent la vidéo | 4h | ⭐⭐ |
| 9 | **Podcast ML** | Audience engagée | 2h | ⭐ |
| 10 | **Product Hunt** | Produit + communauté produit | 2h | ⭐ |

---

## Canaux alternatifs / bonus

| Canal | Pour qui | Pourquoi |
|---|---|---|
| **Substack / newsletter perso** | Si tu as déjà une audience email | Fidélisation |
| **Hugging Face Spaces** | Projets ML/DL | Démo interactive gratuite |
| **Kaggle** | Projets Data Science / ML | Crédibilité technique |
| **GitHub Trending** | Tout projet OSS | Trafic organique massif |
| **Lobsters** (lobste.rs) | Alternative HN technique | Bon si le HN est saturé |
| **Hacker News "Ask HN"** | Pour poser une question précise | Feedback + visibilité |
| **Twitter Spaces / X Spaces** | Si tu as déjà 500+ followers | Discussion en direct |
| **Devfolio / Devpost** | Hackathons | Si le projet a un volet hackathon |
| **Papers with Code** | Projets académiques | Si lié à un papier |
| **Hugging Face Models** | Si tu publies un modèle | Découverte organique |

---

## Métriques de suivi

| Métrique | Cible (succès) | Outil |
|---|---|---|
| GitHub stars | +50 (semaine 1) | GitHub |
| PyPI downloads | +200 (semaine 1) | PyPI stats |
| Comments HN | 20+ | HN |
| Emails capturés | 10+ | Landing page |
| Issues/PR ouvertes | 5+ | GitHub |
| Mentions X | 5+ | X search |

---

## Fichiers associés (à créer avant lancement)
- `LAUNCH_POSTS.md` — drafts Show HN + X + Reddit + Discord
- `docs/hn_feedback_log.md` — log des retours HN
- `docs/launch_plan_{PROJECT}.md` — ce planning
- `docs/launch_postmortem.md` — analyse post-lancement (si échec)
- `docs/tracking/acquisition_tracker.md` — historique des posts et résultats