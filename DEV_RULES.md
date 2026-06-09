# DEV_RULES.md — Règles spécifiques au développement NeuralDBG

> Ces règles s'ajoutent aux kuro-rules générales.
> Elles sont LIES au code, au repo, et au processus de résolution de bugs.
> Lues obligatoirement en début de session avec R1.

---

## Règle D1 : Chaque bug DOIT améliorer NeuralDBG

**Problème** : On documente des bugs sans que NeuralDBG évolue. Zéro valeur ajoutée.

**Solution** : Pour CHAQUE bug chassé, AU MOINS une de ces livrables doit exister :
1. Nouveau type d'événement dans le moteur (ex: `gradient_norm_spike`, `sdpa_fallback`)
2. Nouvelle règle dans `remediation_rules.py` (Neural-Agent)
3. Nouveau template dans `explain.py` (hypothèse causale)
4. Nouveau test dans `tests/` qui valide la détection
5. Amélioration d'un hook existant (ex: composite hook après BUG-001)

**Vérification** :
```
AVANT de documenter un bug:
  -> Qu'est-ce que NeuralDBG gagne ?
  -> Si rien: ne pas créer le bug tracker, retourner au code
```

**Enforcement** : IF bug documented sans amélioration NeuralDBG -> SUPPRIMER le bug tracker.

---

## Règle D2 : Pas de workaround. Jamais.

**Problème** : On écrit "workaround" au lieu de "fix". Un workaround n'est pas une résolution.

**Solution** :
- Si le bug est dans PyTorch/upstream → soumettre un PR qui le RÉSOUT (pas `warnings.warn`)
- Si le bug est dans notre code → le corriger dans NeuralDBG/Neural-Agent
- Si on ne peut pas résoudre → documenter POURQUOI on ne peut pas, puis passer à un autre bug
- Le mot "workaround" est INTERDIT dans les livrables. Utiliser "fix" ou "resolution".

**Vérification** :
```
grep -r "workaround" docs/ examples/ --include="*.md" --include="*.py"
# Doit retourner 0 résultats
```

---

## Règle D3 : Reproduction SANS hardware spécifique

**Problème** : BUG-003 (MPS) et BUG-004 (GPU) nécessitent du hardware qu'on n'a pas. On les documente et on stagne.

**Solution** : Pour chaque bug nécessitant du hardware absent :

| Hardware manquant | Alternative |
|-------------------|-------------|
| GPU CUDA | Kaggle free (T4), Google Colab free (T4), `torch.compile` mode CPU avec simulation |
| MPS (Apple Silicon) | Unit test sur le CODE PATH (pas le hardware) : mocker `torch.device("mps")`, tester la logique de gradient scaling |
| Multi-GPU | `torchrun --nproc_per_node=1` + gradient accumulation simulation |
| Modèle gros (72B) | Utiliser le plus petit modèle dispo (0.6B, 1.5B) qui reproduce le même pattern |

**Pour MPS specifiquement** :
- Lire le code source de `aten/src/ATen/native/mps/operations/` pour comprendre le path
- Écrire un test qui valide le behavior attendu SANS exécuter sur MPS
- Si le bug est numerical (mauvais gradients), reproduire la condition numerique sur CPU

**Enforcement** : IF bug ne peut pas être reproduit -> écrire le test unitaire qui CATCHERAIT le bug si le hardware était disponible, PUIS passer au bug suivant.

---

## Règle D4 : PRs upstream = pipeline complet NeuralDBG + Neural-Agent

**Problème** : PR #186631 (pytorch) fermée car c'était juste un `warnings.warn()`. Pas de valeur NeuralSuite.

**Solution** : Chaque PR upstream doit contenir :
1. **Detection** : montrer que NeuralDBG détecte le bug (ex: output de `explain_failure()`)
2. **Resolution** : montrer que Neural-Agent propose/applique le fix
3. **Preuve** : script de reproduction + log NeuralDBG avant/après fix
4. **Code fix** : le vrai patch dans le code upstream

**Template** : `.github/PR_TEMPLATES/upstream-fix.md`

**Enforcement** : IF PR upstream sans démo NeuralDBG+Agent -> NE PAS SOUMETTRE.

---

## Règle D5 : Les bugs sont des features, pas de la documentation

**Problème** : On crée des fichiers `BUG-XXX.md` qui sont juste des descriptions de bugs. Aucune valeur pour NeuralDBG.

**Solution** : Chaque BUG-XXX.md doit contenir une section "NeuralDBG Improvement" qui détaille EXACTEMENT ce qui a changé dans le code :
- Fichier modifié + ligne
- Nouveau test ajouté
- Nouveau type d'événement
- Nouvelle hypothèse causale

**Enforcement** : IF BUG-XXX.md sans section "NeuralDBG Improvement" avec code reference -> supprimer le fichier.

---

## Règle D6 : Alternatives hardware pour chaque bug

### BUG-003 (MPS wrong gradients) — Plan de reproduction

Le bug : PyTorch MPS retourne des gradients incorrects (pytorch#177116).

**Pas besoin de hardware MPS pour** :
1. Lire le code source PyTorch : `aten/src/ATen/native/mps/operations/Linear.mm`
2. Écrire un test qui compare le gradient CPU vs la valeur attendue
3. Montrer que NeuralDBG detecterait le gradient incorrect via `gradient_health_transition`
4. Écrire la règle Neural-Agent qui suggererait "use CPU for gradient verification"

**Action concrète** :
```python
# test_mps_gradient_detection.py
# Test que NeuralDBG detecte un gradient incorrect PEU IMPORTE le device
def test_gradient_injection_detected():
    """Simule le bug MPS en injectant un gradient incorrect."""
    model = nn.Linear(10, 5)
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    # Injecter le comportement MPS: gradient *= 0 (ou gradient = random)
    with torch.no_grad():
        model.weight.grad.fill_(0.0)  # simulation gradient zero
    # NeuralDBG doit detecter ça
    with NeuralDbg(model) as dbg:
        # re-forward pour capturer
        ...
```

### BUG-004 (Qwen3.5 SDPA) — Plan de reproduction

Le bug : SDPA dense mask → Math backend → BF16 collapse → gradient explosion.

**Pas besoin de GPU A100 pour** :
1. Utiliser `Qwen/Qwen3-0.6B` (600M params, tourne sur T4 Colab free)
2. Kaggle free (16h GPU/mois) — notebook déjà créé
3. Google Colab free (T4 GPU, 4h/session)
4. CPU : forcer SDPA sur petit modèle, vérifier le code path du mask

**Action concrète** :
- Kaggle notebook déjà prêt (`notebooks/train_neuralagent_kaggle.ipynb`)
- Colab : même notebook, upload et exécuter
- Vérifier que `attn_implementation="sdpa"` est bien utilisé

---

## Règle D7 : Checklist avant de dire "bug documenté"

Pour chaque bug, AVANT de marquer comme "done" :

- [ ] Script de reproduction créé et **testé** (pas juste écrit)
- [ ] NeuralDBG amélioré (nouveau code, pas juste docs)
- [ ] Test unitaire ajouté
- [ ] Si hardware manquant : test unitaire qui catcherait le bug
- [ ] PR upstream rédigée (pas juste un commentaire)
- [ ] Aucun "workaround" dans les livrables

---

## Règle D8 : Reproduction distante (Remote Reproducer)

**Problème** : 60%+ des data scientists travaillent sur CPU. Les bugs GPU/CUDA/MPS sont impossibles à reproduire localement. NeuralDBG devient inutile pour ces users.

**Solution** : Module `neuraldbg.remote` qui envoie le script de reproduction vers un service GPU distant :
- Phase 1 : Google Colab (gratuit, T4, upload manuel) — 2 semaines
- Phase 2 : Kaggle (gratuit, 30h/mois, API automatique) — 2 semaines
- Phase 3 : RunPod/Lambda (payant, A100, REST API) — 1 mois

**Architecture** : `docs/REMOTE_REPRODUCE.md`

**Impact marché** : TAM élargi de 40% (GPU users) à 100% (tous les users). Aucun outil de diagnostic ne fait ça.

**Enforcement** : Ne PAS commencer l'implémentation tant que l'architecture n'est pas validée par CEO.

---

**Créé** : 2026-06-08
**Trigger** : 4 bugs documentés, 1 seul a amélioré NeuralDBG (BUG-001), 0 résolus, 0 PRs soumises
**Enforcement** : OBLIGATOIRE
