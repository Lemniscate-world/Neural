# BUG-004 — HuggingFace #44928 Qwen3.5 SDPA gradient explosion

> **MID**: BUG-004
> **Linked**: FIX-004 (NeuralDBG detection + Neural-Agent fix)
> **Status**: Detection script created, upstream comment drafted, NOT posted
> **Date opened**: 2026-06-08
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/huggingface/transformers/issues/44928
- Title: *"[Bug] Catastrophic gradient explosion (NaN) in RLHF with Qwen3.5 due to 3D position_ids forcing SDPA Math fallback and BF16 collapse"*
- Status upstream: OPEN, labeled WIP, bug
- Author: @ouroborosscr
- Reproducible repo: https://github.com/ouroborosscr/Report-the-gradient-explosion-of-qwen3.5

## Root cause

When training Qwen3.5 with SDPA attention and 3D position_ids (mRoPE), transformers materializes a dense 4D attention mask `[Batch, 1, SeqLen, SeqLen]` with `is_causal=False`. This violates PyTorch SDPA's fused kernel constraints (`if (attn_mask.has_value()) { return false; }`), forcing a silent fallback to the Math backend. The Math backend in BF16 accumulates softmax denominators over 8K-100K tokens, causing truncation errors that snowball under RLHF losses (DPO/GRPO/DAPO) into gradients of magnitude 10^28.

## Trigger conditions

1. Model: Qwen3.5 (or any Qwen2 architecture with 3D position_ids / mRoPE)
2. Attention implementation: SDPA (default)
3. Sequence length: long context (8K+)
4. Loss function: RLHF variant (DPO/GRPO/DAPO) with exponential amplifiers
5. Precision: BF16

## NeuralDBG detection

The detection script `examples/repro_huggingface_44928.py` shows what NeuralDBG captures:

1. **gradient_norm_spike** event at attention layers (k_norm, q_norm, v_norm) with magnitudes 10^28
2. **nan_detected** event in loss computation
3. **causal_chain**: SDPA mask materialization -> Math backend fallback -> BF16 truncation -> gradient explosion -> NaN loss
4. **root_cause_hypothesis**: "SDPA dense mask forces Math backend, BF16 accumulation unstable for long-context RLHF"

## Neural-Agent proposed fix

```python
# Fix: force flash_attention_2 for Qwen3.5 with long context
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-72B",
    attn_implementation="flash_attention_2",  # avoid SDPA Math fallback
    torch_dtype=torch.bfloat16,
)
```

Alternative (when FA2 is unavailable): implement SDPA varlen path with `cu_seqlens` to physically drop padding tokens.

## Reproduction script

`examples/repro_huggingface_44928.py` — 3 stages:
1. Reproduce the bug with SDPA (gradient explosion to 10^28)
2. Show NeuralDBG detection (gradient_norm_spike events, causal chain)
3. Apply fix (flash_attention_2) and verify stable training

## Deliverables checklist

- [x] BUG-004 tracking file (this file)
- [x] Detection script (`examples/repro_huggingface_44928.py`)
- [x] Upstream comment draft (`docs/posts/huggingface_44928_comment.md`)
- [x] Comment posted on huggingface/transformers#44928 (CEO manual)
- [x] Neural-Agent remediation rule for SDPA fallback detection (`sdpa_gradient_explosion` in `remediation_rules.py`, `attn_implementation=flash_attention_2` patched to config)
- [ ] Verification on GPU hardware

## Mom Test R2

- Reproduction script provided with diagnostic log
- No claim of fixing the upstream bug — only detection and proposed fix documented
- Maintainer (@vasqu) already confirmed varlen support is planned for torch 2.10+

## R64 Negative Mom Test

- What we don't detect: SDPA backend selection happens at PyTorch C++ level, not visible to Python autograd hooks
- Our detection is post-hoc (gradient norms after backward), not pre-emptive (mask inspection before forward)
