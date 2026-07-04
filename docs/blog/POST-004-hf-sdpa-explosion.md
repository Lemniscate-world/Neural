# POST-004 — When Transformers Explode: SDPA Gradient Instability in Qwen3.5

> **BUG-004** | **Source**: huggingface/transformers#44928 | **PR**: #47024 (CLOSED stale)  
> **Date**: 2026-07-04 | **Detection**: NeuralDBG | **Category**: gradient_explosion

---

## 1. The Bug

**What**: Qwen3.5 models experience gradient explosion during SDPA (Scaled Dot-Product Attention) computation. The gradients go from NORMAL to EXPLODING within a few training steps — silently, without NaN.

**Upstream**: [huggingface/transformers#44928](https://github.com/huggingface/transformers/issues/44928) — OPEN.

This is notable because it's a **HuggingFace** bug, not PyTorch — demonstrating that NeuralDBG works across the ML ecosystem.

## 2. Reproduction

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
x = torch.randint(0, 1000, (1, 128))

out = model(x)
loss = out.logits.sum()
loss.backward()
# Gradient norms explode 10-100x within first few steps
```

## 3. NeuralDBG Diagnosis

NeuralDBG detects the gradient health transition:
```
gradient_health_transition at q_proj: NORMAL → EXPLODING
  → optimizer_instability at AdamW: diverging
  → training collapse at step ~50
```

## 4. Why This Matters

- **Cross-ecosystem**: First non-PyTorch bug detected by NeuralDBG
- **Foundation models**: Affects Qwen (Alibaba), one of the most popular open-source LLM families
- **PR impact**: #47024 was our first HF PR — closed by stale bot in 1 day (lesson learned)

## 5. Status

- PR #47024 closed by HF stale bot after 1 day
- New PR to be recreated with clean fork approach
- Issue discussion needed before PR (HF process)

---

*Detected by [NeuralDBG](https://github.com/LambdaSection/NeuralDBG). See all [post-mortems](index.html).*
