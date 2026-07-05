# PLAN.md -- NeuralDBG Strategic Plan

> Last Updated: 2026-07-05 13:00 — BLACK-SWAN STRATEGY ACTIVE.
> **Doctrine**: "Expect the worst. Detect the non-existent. +10 resilience everywhere."
> **Long-term**: Never forget — we prepare for failures nobody has seen yet.

---

## Dashboard — 5 Juillet 2026

| Pilier | Status |
|--------|--------|
| Detection FF (MLP/CNN/TF) | 🟢 93% |
| Detection RNN | 🟡 68% |
| Detection Hybrid | 🟢 96% |
| Detection Global (quick 50) | 🟢 90% |
| Vanishing detection | 🟡 67% full / 88% quick |
| **Black-Swan catalog** | 🟢 50+ untested archs identified |
| **+10 Resilience target** | 🟡 Per-module stress tests pending |
| Paper scraper auto | 🟡 To build |
| GPU v4 | 🟢 92.3% |
| Aquarium | 🟢 Live |
| Paper draft | 🟢 Complete |
| PRs | 🔴 4 actives, 0 merges |

---

## BLACK-SWAN STRATEGY (Long-term, never forget)

### Pourquoi
Les vrais bugs de production ne sont pas les 6 qu on teste.
Ce sont les interactions inattendues, le hardware bizarre, les archis jamais vues.
NeuralSuite doit détecter ce qui n existe pas encore.

### Tiers d attaque

| Tier | Nom | Action |
|------|-----|--------|
| 1 | **Connus inconnus** | Tester GNN, MoE, Diffusion, FlashAttention, Quantized, Federated |
| 2 | **Inconnus inconnus** | Fuzzer d architectures, injection adversariale, stress test valeurs extremes |
| 3 | **Prédictif** | Anomaly detector non-supervisé sur dynamiques d entraînement normales |

### +10 Resilience — Chaque module doit tenir

- 10x gradient normal → pas de NaN
- 0.1x gradient normal → détection vanishing
- 10x input scale → pas d explosion
- NaN/Inf n importe ou → détecté et localisé
- Mixed precision → pas d underflow silencieux
- 100K+ tokens → softmax stable
- 10K+ couches → gradient flow intact

### Architectures restantes à tester (50+)

GNN, FlashAttention, MoE+load balancing, Diffusion/UNet, Neural ODE,
Quantized (INT8/INT4/GPTQ/AWQ), Multi-modal (CLIP/LLaVA), RAG,
Actor-Critic/DQN, Federated Learning, Normalizer-Free, Mixture-of-LoRA,
Sparsely-gated, Continuous-depth, Hardware-specific (MPS/XLA/CUDA graph)

### Black-Swan par catégorie

| Catégorie | Exemples |
|-----------|----------|
| Architecturale | GNN message passing, MoE dead experts, FlashAttention hooks |
| Hardware | MPS gradient corruption, XLA reordering, fp16 underflow |
| Numérique | LayerNorm cancellation, softmax saturation, einsum silent |
| Interaction | GradClip+LayerNorm, WeightDecay+AdamW+fp16, compile+autograd |

---

## Progres par Bloc

| Bloc | Status | Reste |
|------|--------|-------|
| A (Contenu) | 🟢 100% | — |
| B (Produit) | 🟢 100% | — |
| C (Distribution) | 🟡 66% | Poster manuellement |
| D (PRs) | 🔴 Suivi | Relance manuelle |
| E (Black-Swan) | 🟡 20% | Automated scraper, GNN/MoE/Diffusion testers, fuzzer |

---

## Prochaines actions immédiates

1. Build automated paper scraper (arxiv API daily)
2. Add GNN + MoE + Diffusion to combinatorial tester
3. Build architecture fuzzer (random valid model generator)
4. Stress test suite: extreme values on all modules
5. Full 200-arch sweep with family threshold (get definitive 82-85%)
