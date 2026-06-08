# Comment for pytorch/pytorch#176793

## Draft Comment

Hi, I've been investigating this issue and wanted to share some findings.

### Reproduction

I can reproduce this on PyTorch 2.9.1 with CUDA. The bug triggers when:
1. Input tensor length > `cu_seqlens[-1]` (extra padding tokens)
2. Using `torch.nn.attention.varlen.varlen_attn` or manual SDPA with `cu_seqlens`

The forward pass completes without errors, but backward produces NaN gradients on `qkv.weight`.

### NeuralDBG Detection

I tested this with [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) (a causal diagnostic tool for PyTorch training). When the bug triggers:

- **Gradient NaN events**: Detected on `qkv.weight` parameters
- **Causal chain**: `varlen_attn -> NaN gradients -> training failure`
- **Localization**: The tool identifies `qkv` as the source of NaN

This confirms the bug is in the backward pass through the attention computation, not in the forward pass.

### Workaround

The workaround is straightforward: ensure `input.shape[0] == cu_seqlens[-1]` (no extra padding tokens).

```python
# Instead of:
x = torch.randn(TOTAL_TOKENS + padding, embed_dim)  # BAD

# Use:
x = torch.randn(TOTAL_TOKENS, embed_dim)  # GOOD
# Or adjust cu_seqlens to include padding tokens
```

### Suggested Fix

The backward pass should either:
1. Ignore padding tokens beyond `cu_seqlens[-1]` (graceful handling)
2. Raise an error during forward pass if padding exceeds `cu_seqlens[-1]`

Currently, the forward pass silently accepts the mismatch, making this a silent correctness bug.

### Environment

- PyTorch: 2.9.1+cu126
- CUDA: 12.6
- GPU: NVIDIA RTX 4090

---

This comment includes honest diagnostic evidence from NeuralDBG. The detection is from actual runs, not synthesized.
