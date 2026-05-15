# RULE 94: Daily X Post — Obligation de Publication Quotidienne

Trigger: à la fin de CHAQUE session de travail (obligatoire, comme R83).

---

## Purpose

Maintenir une présence publique quotidienne sur X (anciennement Twitter).
Chaque session de travail doit produire 1 post publiable, peu importe la taille
de la session.

---

## Format

Un post unique, max 280 caractères, formaté ainsi :

```
[emoji] NeuralDBG: [résultat clé de la session]

[une métrique ou un chiffre fort]

[#NeuralDBG #ML #DeepLearning]
```

---

## Règles de contenu

1. **Ne JAMAIS** révéler le code propriétaire, les heuristiques causales, ou les
   détails implémentatoires de NeuralDBG-Engine
2. **Toujours** inclure une métrique chiffrée (ex: "135 tests pass", "0.905 accuracy",
   "5 architectures dogfoodées", etc.)
3. **Toujours** mentionner un livrable concret (demo, benchmark, test, etc.)
4. **Jamais** de promesses sur des dates ou features futures
5. **Ton** : technique, factuel, pas de hype
6. **Hashtags** : minimum 2, maximum 3 (#NeuralDBG + domaine concerné)

---

## Exemple

```
NeuralDBG: LoRA fine-tuning dogfooding done — 3 failure scenarios
(NaN, exploding, catastrophic forgetting) now tracked.

138 tests pass, 6 architectures couvertes.

#NeuralDBG #LLM #FineTuning
```

---

## Emplacement

- Le post est sauvegardé dans `outputs/x_post_YYYY-MM-DD.md`
- Le fichier contient uniquement le texte du post (pas de métadonnées)
- Prêt à être copié-collé sur X

---

## Sanction

Si une session se termine sans post X (ou sans raison valide documentée),
l'agent DOIT le signaler comme tâche non complétée dans le résumé de session.
