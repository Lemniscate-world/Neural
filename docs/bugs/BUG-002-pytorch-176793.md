# BUG-002 — PyTorch #176793 NaN gradients in varlen_attn with padding

> **MID**: BUG-002
> **Status**: Open upstream — reproduction confirmed
> **Date opened**: 2026-06-08
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/176793
- Title: *"NaN gradients in varlen_attn backward pass when input length exceeds cu_seqlens[-1]"*
- Status upstream: OPEN, since 2026-03-07
- Labels: module: autograd, module: nn, module: cuda, module: correctness (silent), module: sdpa

## Trigger conditions

When using `torch.nn.attention.varlen.varlen_attn`, padding the input tensor so that its total length is greater than the total number of tokens defined by `cu_seqlens[-1]` causes NaN gradients during the backward pass.

The forward pass executes without raising any shape mismatch or out-of-bounds errors. Only the backward pass produces NaN values.

Minimal repro (from upstream):

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask

device = "cuda"
TOTAL_TOKENS = 944
cu_seqlens = torch.tensor([0, 144, 432, 944], dtype=torch.int32, device=device)
max_seqlen = 512

# Add 2 padding tokens -> triggers NaN
x = torch.randn(TOTAL_TOKENS + 2, 1024, device=device, requires_grad=True)

qkv = torch.nn.Linear(1024, 3072, device=device)
out = torch.nn.Linear(1024, 1024, device=device)

with torch.autocast(device):
    q, k, v = qkv(x).chunk(3, dim=-1)
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False
    )
    loss = out(attn_out)[:cu_seqlens[-1]].abs().sum()
    loss.backward()

for name, param in qkv.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN detected in gradients for {name}!")
        break
# Output: NaN detected in gradients for weight!
```

## Why this matters for NeuralDBG

This bug is relevant to NeuralDBG because:

1. **Silent failure**: Forward pass is correct, only backward produces NaN
2. **Padding-related**: Common pattern in real-world training (variable-length sequences)
3. **Similar to BUG-001**: Both involve attention mechanisms with masking/padding
4. **Hard to detect**: Loss may be finite (if NaN row is masked out of loss computation)

## Relationship to BUG-001

BUG-001 (pytorch#41508) involved `nn.MultiheadAttention` with fully masked rows. BUG-002 involves `varlen_attn` with padding beyond `cu_seqlens[-1]`. Both are attention-related bugs that produce NaN gradients, but through different mechanisms:

| Aspect | BUG-001 | BUG-002 |
|--------|---------|---------|
| Module | `nn.MultiheadAttention` | `varlen_attn` (flex attention) |
| Trigger | Fully masked row in attn_mask | Padding beyond cu_seqlens |
| Forward | Correct | Correct |
| Backward | NaN in in_proj_weight | NaN in qkv.weight |
| Root cause | Composite module blind spot | Padding token handling |

## Potential NeuralDBG detection

NeuralDBG should detect this via:
1. Gradient NaN events on `qkv.weight` parameters
2. Causal chain: varlen_attn -> NaN gradients -> training failure
3. Localization: qkv layer is the source of NaN

## Deliverables checklist

- [ ] Reproduction script (`examples/repro_pytorch_176793.py`)
- [ ] BUG-002 tracking file (this file)
- [ ] NeuralDBG detection confirmed
- [ ] Comment posted on pytorch/pytorch#176793 with link to detection
- [ ] Postmortem blog (if pattern is interesting enough)
- [ ] NeuralAgent remediation rule (pad sequences to avoid padding beyond cu_seqlens)

## Sign-off

- Mom Test R2: reproduction script included. No claim of fixing the upstream bug — only detection and documentation are owned.
- R64 Negative Mom Test: what we detect is documented. What we don't (e.g., specific cuDNN backend issues) is acknowledged.
