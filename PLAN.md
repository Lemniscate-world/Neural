# PLAN.md -- NeuralDBG Strategic Plan

> Last Updated: 2026-07-04 19:30 -- Sweep 87%, v4 dataset 538 ex, GPU training launched.

---

## Dashboard -- 4 Juillet 2026 Final

| Pilier | Status | Nous controle ? |
|--------|--------|:---:|
| Detection FF (MLP/CNN/TF) | 🟢 93% | Oui |
| Detection RNN | 🟢 65% (was 49%, +16%) | Oui |
| Detection Hybrid | 🟢 85% (was 34%, +51%) | Oui |
| Detection Global | 🟢 87% (was 75%, +12%) | Oui |
| Causal Chains | 🟢 Operationnel | Oui |
| GPU Model v3 | 🟢 5/5 categories | Oui |
| GPU Model v4 | 🟡 538 ex, training en cours | Oui |
| Pipeline E2E | 🟢 Prouve | Oui |
| Paper Archs | 🟢 60 + scraper | Oui |
| Engine | 🟢 Merge dans core | Oui |
| Blog + Posts | 🟢 Complete | Oui |
| PRs | 🔴 4 actives, 0 merges | Non |
| Stars | 🔴 24 | Non |

**Principe : 80% sur ce qu on controle. PRs et stars suivront.**

---

## Ecosystem Simplified

| Repo | Status | Role |
|------|--------|------|
| NeuralDBG (public, MIT) | 🟢 v1.4.0-dev | Engine merged, RNN fix, 87% detection |
| Neural-Agent (private) | 🟢 Active | v4 training en cours. Monetizable. |
| Aquarium (public) | 🟡 Dormant | Reboot as web dashboard. |

---

## Progres par Bloc

| Bloc | Status | Reste a faire |
|------|--------|---------------|
| A (Contenu) | 🟢 90% | Video demo (A3) |
| B (Produit) | 🟢 100% | -- |
| C (Distribution) | 🟡 66% | Poster PTD + Reddit |
| D (PRs) | 🔴 Suivi | Relance J+3: 5 Juillet |
| GPU v4 | 🟡 Training | 538 ex, fp16, Qwen2-0.5B + LoRA |

---

## Boucle d amelioration (4 Juillet)

```
TESTER 200 archs -> trouve RNN 49%, Hybrid 34%
        |
FIX    isinstance(output, Tensor) skip les RNN -> unwrap + hidden state
        |
RESULT 87% global (+12%), RNN 65% (+16%), Hybrid 85% (+51%)
        |
DATA   450 exemples x 5 families -> merge -> 538 total (6.1x original)
        |
MODEL  GPU training v4 lance (Qwen2-0.5B + LoRA, fp16, 3 epochs)
```

## Prochaines actions

1. Terminer GPU training v4 -> benchmark agent sur RNN
2. Poster PTD + Reddit (contenu pret)
3. Relance PRs demain (5 Juillet)
4. Aquarium web dashboard
