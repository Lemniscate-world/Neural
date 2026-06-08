# BUG-003 — PyTorch #177116 — MPS catastrophically wrong gradients

> **MID**: BUG-003
> **Status**: OPEN upstream — high priority
> **Date opened**: 2026-06-08
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/177116
- Title: *"MPS: catastrophically wrong gradients in backward pass (>32K elements)"*
- Status upstream: OPEN, since 2026-03-11
- Labels: high priority, module: autograd, module: mps, module: correctness (silent)

## Trigger conditions

The MPS backend produces catastrophically wrong gradients (1,000x to 100,000x too large) during `loss.backward()` when ALL of the following are true:

1. A prior MPS forward+backward pass has occurred in the same process at a **different batch size**
2. The total number of elements (batch_size x seq_len) exceeds **~32,768 (2^15)**
3. The loss function involves complex backward operations (CrossEntropyLoss, MSELoss)

The forward pass is ALWAYS correct. Only the backward pass is affected.

Minimal repro (from upstream):

```python
import torch
import torch.nn as nn

class ResidualModel(nn.Module):
    def __init__(self, d=512, V=1000):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, V)

    def forward(self, x):
        h = self.embed(x)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)

# Step 1: Prime the bug with a DIFFERENT batch size
x_prime = torch.randint(0, 1000, (1024, 16)).to("mps")
model = ResidualModel().to("mps")
criterion = nn.CrossEntropyLoss()
loss = criterion(model(x_prime).view(-1, 1000), x_prime.view(-1))
loss.backward()  # This is CORRECT

# Step 2: Now use the target batch size (>32K elements)
for trial in range(5):
    torch.manual_seed(0)
    model = ResidualModel().to("mps")
    x = torch.randint(0, 1000, (4097, 8)).to("mps")  # 32,776 elements
    loss = criterion(model(x).view(-1, 1000), x.view(-1))
    loss.backward()
    gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5
    print(f"  trial {trial}: loss={loss.item():.6f}  grad_norm={gnorm:.4f}")
    # Loss is always correct (~5.09)
    # But grad_norm jumps from 0.24 to 3529 to 16290 (!)
```

Key observations from upstream:
- **Threshold near 2^15 elements**, varies between process invocations
- **Loss always correct** (forward pass not affected)
- **Gradient norms wrong by 1,000x to 68,000x** on subsequent trials
- **`torch.mps.empty_cache()` reduces but doesn't eliminate** the bug
- A VAE encoder training completely failed (loss stuck at 0.55 for 80 epochs)

## Why this matters for NeuralDBG

This is the EXACT failure mode NeuralDBG is designed to detect:

1. **Silent gradient corruption**: Forward pass produces valid loss, backward produces garbage gradients
2. **Layer-level localization**: NeuralDBG hooks would detect gradient norm anomalies on specific layers (fc1.weight, fc2.weight)
3. **Causal chain**: MPS buffer pool corruption -> wrong gradients -> no learning -> stuck loss

If a user ran NeuralDBG on this bug, the output would show:
- `gradient_health_transition` event: gradient norms explode from ~0.24 to ~3500
- Hypothesis: "Gradient explosion detected on Linear layers"
- Localization: fc1.weight, fc2.weight

## Relationship to other bugs

| Aspect | BUG-001 | BUG-002 | BUG-003 |
|--------|---------|---------|---------|
| Module | nn.MultiheadAttention | varlen_attn | MPS backend |
| Trigger | Fully masked row | Padding beyond cu_seqlens | Buffer pool reuse |
| Forward | Correct | Correct | Correct |
| Backward | NaN gradients | NaN gradients | Wrong magnitude gradients |
| Root cause | Composite module blind spot | Padding handling | MPS buffer corruption |
| Severity | High | High | Critical (100Kx wrong) |

## Workaround

- Use CPU or CUDA backend instead of MPS
- Keep batch size fixed (don't change between trials)
- Upgrade to PyTorch >= 2.11.0 (may fix some cases)

## Deliverables checklist

- [x] BUG-003 tracking file (this file)
- [ ] Reproduction script (`examples/repro_pytorch_177116.py`)
- [ ] NeuralDBG detection confirmed (requires MPS hardware)
- [ ] Comment posted on pytorch/pytorch#177116 with link to detection
- [ ] Benchmark scenario for MPS gradient corruption

## Sign-off

- Mom Test R2: reproduction script from upstream included. No claim of fixing the upstream bug.
- R64 Negative Mom Test: we acknowledge this bug may be fixed in newer PyTorch versions (2.11.0+). Our detection capability is what we document, not the bug fix.
