# REMOTE_REPRODUCE.md — Architecture du reproduceur distant

> MID: REMOTE-001
> Status: PLAN — pas encore implémenté
> Date: 2026-06-08

## Problème

Beaucoup de bugs deep learning ne se reproduisent que sur du hardware spécifique :
- GPU CUDA (bugs SDPA, cuDNN, FlashAttention)
- MPS Apple Silicon (bugs de gradient)
- Multi-GPU (bugs FSDP, DeepSpeed, DDP)
- GPU gros (A100/H100 pour bugs de mémoire)

Si l'utilisateur n'a pas ce hardware, il ne peut pas reproduire le bug. NeuralDBG devient inutile.

## Solution : Remote Reproducer

Un module NeuralDBG qui envoie automatiquement le script de reproduction vers un service GPU distant, exécute le test, et récupère les résultats.

## Architecture

```
User machine (CPU)          Cloud GPU (T4/A100)
┌─────────────────┐         ┌─────────────────┐
│ NeuralDBG       │         │                 │
│   reproduce()   │ ──────> │ Script executed │
│                 │  API    │ Results logged   │
│ Results parsed  │ <────── │ NeuralDBG events │
└─────────────────┘         └─────────────────┘
```

## Providers cibles (par priorité)

### Tier 1 — Gratuits / freemium
| Provider | GPU | Temps gratuit | API | Difficulté integration |
|----------|-----|---------------|-----|------------------------|
| Google Colab | T4 (16GB) | 4h/session, illimité | `google-colab` SDK | Facile |
| Kaggle | T4 (16GB) | 30h/mois | Kaggle API | Facile |

### Tier 2 — Payants (cheap)
| Provider | GPU | Prix/heure | API | Difficulté |
|----------|-----|------------|-----|------------|
| RunPod | A100 40GB | ~$1.10/h | REST API | Moyen |
| Lambda Cloud | A100 80GB | ~$1.10/h | REST API | Moyen |
| Vast.ai | Variable | ~$0.20/h | REST API | Difficile |

### Tier 3 — Enterprise
| Provider | GPU | Prix | API |
|----------|-----|------|-----|
| AWS SageMaker | Various | Variable | boto3 |
| GCP Vertex AI | Various | Variable | gcloud SDK |

## Interface Python

```python
from neuraldbg import NeuralDbg
from neuraldbg.remote import RemoteReproducer

# Crée le script de reproduction
with NeuralDbg(model) as dbg:
    # ... forward/backward ...
    script = dbg.create_reproduction_script(
        bug_description="NaN gradients in varlen_attn with padding",
        trigger_conditions={"cuda": True, "min_memory": "16GB"},
    )

# Exécute à distance
reproducer = RemoteReproducer(
    provider="colab",  # ou "kaggle", "runpod", "lambda"
    gpu_type="T4",     # ou "A100", "V100"
    timeout=600,       # 10 minutes max
)

result = reproducer.run(script)
# result.events = [SemanticEvent, ...]
# result.hypotheses = [CausalHypothesis, ...]
# result.log = "full stdout/stderr"
# result.cost = 0.0  # Colab gratuit
```

## flux de travail

1. **Utilisateur** : "J'ai un bug avec mon modèle sur GPU"
2. **NeuralDBG** : génère le script de reproduction
3. **RemoteReproducer** : choisit le meilleur provider (gratuit d'abord)
4. **Upload** : script + dépendances vers le provider
5. **Exécution** : GPU distant exécute le script avec NeuralDBG embarqué
6. **Récupération** : résultats JSON (events, hypothèses, logs)
7. **Affichage** : NeuralDBG présente les résultats comme si le bug avait été reproduit localement

## Défis techniques

### 1. Installation de NeuralDBG sur le remote
```bash
# Le script doit installer NeuralDBG automatiquement
pip install neuraldbg
python -c "from neuraldbg import NeuralDbg; ..."
```

### 2. Transfert des résultats
- Les events JSON sont petits (< 1MB) → transfert facile
- Les tensors sont gros → on ne transfère que les métadonnées (normes, shapes, dtypes)
- Le script remote doit appeler `dbg.export_json()` et écrire le résultat

### 3. Authentification
- Colab : pas d'API key (notebook manuel)
- Kaggle : `~/.kaggle/kaggle.json`
- RunPod/Lambda : API key dans env vars

### 4. Sécurité
- Le script uploadé ne doit PAS contenir de données sensibles
- NeuralDBG ne capture que les métriques d'entraînement, pas les données
- Les résultats sont chiffrés en transit

## Impact marché

### Avant (sans remote)
- Utilisateur sans GPU → "je ne peux pas reproduire" → NeuralDBG inutile
- 60%+ des data scientists travaillent sur CPU (dev) avec GPU limité

### Après (avec remote)
- Utilisateur sans GPU → NeuralDBG reproduit automatiquement → valeur immédiate
- **Différenciateur unique** : aucun outil de diagnostic ne fait ça
- W&B, MLflow, TensorBoard = observabilité passive. NeuralDBG = diagnostic actif avec résolution.

### Chiffres
- TAM élargi : de 40% (GPU users) à 100% (tous les users)
- Coût d'acquisition : réduit (l'outil "fonctionne" même sans GPU)
- Rétention : augmentée (results dans le cloud, partageables)

## MVP (Minimum Viable Product)

### Phase 1 — Colab (2 semaines)
- Script généré automatiquement
- Upload manuel vers Colab notebook
- Résultats parsés depuis le notebook output
- **Pas d'API automatique** (Colab n'a pas d'API pour créer des notebooks)

### Phase 2 — Kaggle (2 semaines)
- API Kaggle pour créer des notebooks
- Exécution automatique via `kaggle kernels push`
- Récupération des résultats via `kaggle kernels output`
- **Gratuit : 30h GPU/mois**

### Phase 3 — RunPod/Lambda (1 mois)
- REST API pour créer des pods
- Upload de script + résultats
- Facturation à l'heure
- **Payant mais puissant**

## Fichiers à créer

```
neuraldbg/
  remote/
    __init__.py
    base.py          # Classe abstraite RemoteReproducer
    colab.py         # Google Colab (upload manuel)
    kaggle.py        # Kaggle API (automatique)
    runpod.py        # RunPod REST API
    lambda_cloud.py  # Lambda Cloud REST API

tests/unit/
  test_remote_reproduce.py

docs/
  REMOTE_REPRODUCE.md  # ce fichier
```
