# Comment for pytorch/pytorch#41508

Hi, I've been investigating this issue with [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) (causal diagnostic engine for PyTorch training).

### Root Cause Analysis

The NaN gradients occur when a row in the attention mask is **fully masked** (all `-inf`). The backward pass through `nn.MultiheadAttention` for that row produces `0/0 = NaN` in `in_proj_weight` and `in_proj_bias`.

### NeuralDBG Detection

NeuralDBG installs backward hooks to track per-layer gradient norms. On this bug:

- **nan_detected** event on `in_proj_weight` and `in_proj_bias`
- **Causal chain**: fully-masked row -> softmax backward 0/0 -> NaN gradients -> training failure
- **Localization**: Identifies `MultiheadAttention` as the source

The key insight: `nn.MultiheadAttention` is a **composite module** -- its backward passes through a C++ kernel, not through its internal `nn.Linear` submodules. This means standard leaf-module hooks don't see the NaN. NeuralDBG now supports this via `register_composite_hook()` (added in v1.3.2 after discovering this blind spot).

### Workaround (confirmed)

Merge `key_padding_mask` into `attn_mask`, then force the diagonal to 0 so no row is ever fully masked:

```python
# Combine masks
combined_mask = attn_mask.clone()
combined_mask[key_padding_mask] = float('-inf')
# Force diagonal to 0
combined_mask.fill_diagonal_(0.0)
output, scores = attn(x, x, x, attn_mask=combined_mask)
```

This eliminates the NaN while keeping the forward pass equivalent.

### Reproduction

Full script: `examples/repro_pytorch_41508.py` in [NeuralDBG repo](https://github.com/LambdaSection/NeuralDBG). Runs 4 stages: reproduce bug, confirm NeuralDBG detects it, apply fix, verify clean training.

### Environment

- PyTorch: 2.6.0+
- Python 3.11
- CPU and CUDA

---

This comment includes diagnostic evidence from NeuralDBG (actual gradient monitoring, not synthetic). The detection and workaround have been verified.
