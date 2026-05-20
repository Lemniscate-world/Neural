# CDP — Causal Diagnostic Protocol

> Proposition de protocole machine-readable pour le diagnostic d'entraînement de réseaux de neurones.
> NeuralDBG en est la première implémentation de référence.

---

## 1. Qu'est-ce qu'un protocole ?

Un protocole, c'est un **format standard** que tout le monde peut utiliser pour parler le même langage.

**Exemples :**
- **HTTP** = protocole pour le web → n'importe quel serveur peut parler HTTP
- **USB** = protocole pour les périphériques → n'importe quelle souris USB marche sur n'importe quel PC
- **CDP** = protocole pour le diagnostic causal → n'importe quel outil peut produire ou consommer un diagnostic CDP

**CDP n'est PAS NeuralDBG.** NeuralDBG est une implémentation de CDP. D'autres outils pourraient aussi produire du CDP.

---

## 2. Fallbacks — est-ce que ça expose notre code ?

**Non.** Les fallbacks sont des heuristiques basiques (seuils, calculs mathématiques simples) qui sont DANS le code public. Elles ne révèlent PAS les heuristiques avancées de l'engine propriétaire.

| Ce que voit l'utilisateur | Ce qui reste privé |
|---|---|
| `if norm < 1e-6: return VANISHING` | Algorithme de classification bayésienne |
| `if dead_ratio > 0.5: return DEAD` | Base de patterns de défaillance (100+ scénarios) |
| `return []` pour coupled_failures | Algorithme de détection de couplages |
| Export JSON basique | Enrichissement causal, confidences avancées |

**C'est comme Firefox (open source) vs le moteur de recherche Google (propriétaire).** Tout le monde peut voir le code de Firefox, ça ne révèle pas comment PageRank marche.

---

## 3. La différence sans engine vs avec engine

| Scénario | Sans engine | Avec engine |
|---|---|---|
| Tu fais un quickstart | ✅ Tu vois des events capturés | ✅ Pareil + hypothèses détaillées |
| Tu detectes un NaN | ✅ Tu vois "NaN detected in layer X" | ✅ Tu sais POURQUOI (LR trop haut, init foireuse) |
| Tu exportes en JSON | ✅ Fichier valide | ✅ Fichier enrichi |
| Tu veux la cause racine | ❌ Liste d'events bruts | ✅ "layer4.0.conv1 à l'étape 234" |
| Tu veux auto-corriger | ❌ Pas possible | ✅ Possible via CDP → agent |

**Sans engine, NeuralDBG est un détecteur d'events. Avec engine, c'est un diagnostiqueur causal.**

---

## 4. Papier de recherche — proposition

Titre : **"CDP: A Causal Diagnostic Protocol for Neural Network Training"**

On peut écrire ce papier après le Show HN, quand on aura du feedback utilisateur pour renforcer les claims. Ça devient un argument de vente puissant : "notre protocole est cité dans un papier".

---

## 5. Prochaine étape

- [ ] Show HN le 26 mai (priorité)
- [ ] Collecter feedback utilisateur
- [ ] Formaliser CDP en spec (semaine suivante)
- [ ] Écrire le papier de recherche (juin)