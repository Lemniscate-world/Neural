# POST-009 — The 32-Bit Time Bomb: SDPA Offset Overflow in Mem-Efficient Attention

> **BUG-009** | **Source**: pytorch/pytorch#187227  
> **Date**: 2026-07-04 | **Detection**: SKIP (shape mismatch on current test harness)

---

## 1. The Bug

**What**: `scaled_dot_product_attention` with mem-efficient backend uses 32-bit integer offsets internally. For very long sequences (> 2^31 tokens), the offset overflows, causing incorrect attention computation. Results are wrong but finite — no NaN, no crash.

**Upstream**: [pytorch#187227](https://github.com/pytorch/pytorch/issues/187227) — OPEN.

## 2. Why It Matters

- **Scale issue**: Becomes relevant as context windows grow (100K → 1M tokens)
- **Silent**: Overflow happens in C++/CUDA internals, invisible to Python monitoring
- **Future-proofing**: As models scale to million-token contexts, this bug becomes critical

## 3. Detection Status

This bug requires specific input shapes that are incompatible with our current test harness — classified as SKIP. Future work: adapt the DeepMLP harness for variable-shape inputs.

---

*Cataloged by [NeuralDBG](https://github.com/LambdaSection/NeuralDBG). See all [post-mortems](index.html).*
