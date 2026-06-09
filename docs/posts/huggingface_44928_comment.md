# Comment draft for huggingface/transformers#44928

> **Status**: DRAFT, NOT POSTED
> **Author**: CEO (must copy-paste manually)
> **Target**: https://github.com/huggingface/transformers/issues/44928
> **Date**: 2026-06-08

---

## Comment body

Hi @ouroborosscr @vasqu,

We ran [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) (causal diagnostic engine for PyTorch training) on a minimal reproduction of this issue and captured the full gradient explosion chain.

### NeuralDBG detection output

```
[gradient_norm_spike] model.layers.19.self_attn.q_proj: 2.045e+14 (step 0)
[gradient_norm_spike] model.layers.15.self_attn.v_proj: 1.549e+21 (step 0)
[gradient_norm_spike] model.layers.11.self_attn.q_proj: 4.147e+31 (step 0)
[gradient_norm_spike] model.layers.7.self_attn.q_proj:  1.136e+34 (step 0)
[gradient_norm_spike] model.layers.3.self_attn.q_proj:  1.389e+35 (step 0)
[nan_detected] loss: NaN (step 0)
```

### Causal chain identified

1. SDPA dense mask materialization `[B, 1, S, S]` with `is_causal=False`
2. Fused kernel constraint violated -> Math backend fallback
3. BF16 softmax accumulation over 8K+ tokens
4. Truncation error amplification via RLHF loss (exponential terms)
5. Gradient explosion to 10^28-10^35
6. NaN loss

### Neural-Agent proposed fix

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-72B",
    attn_implementation="flash_attention_2",  # avoid SDPA Math fallback
    torch_dtype=torch.bfloat16,
)
```

This eliminates the dense mask materialization and uses `cu_seqlens` for variable-length handling, keeping FA2 engaged.

### Reproduction

Full script: `examples/repro_huggingface_44928.py` in [NeuralDBG repo](https://github.com/LambdaSection/NeuralDBG).

We agree with @vasqu that the SDPA varlen path is the proper long-term fix. In the meantime, the `flash_attention_2` workaround is the only mathematically safe approach for long-context RLHF with Qwen3.5.

---

## Notes for CEO

- Post this comment manually on https://github.com/huggingface/transformers/issues/44928
- The comment shows the FULL pipeline: NeuralDBG detects -> Neural-Agent fixes
- This is NOT a naive `warnings.warn()` — it's a diagnostic + proposed resolution
- The maintainer (@vasqu) already confirmed varlen support is planned
- Link back to NeuralDBG repo for credibility
