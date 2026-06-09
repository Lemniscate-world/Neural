# BUG-002 — PyTorch #176793 NaN gradients in varlen_attn with padding

> **MID**: BUG-002
> **Status**: Detection test created, NeuralDBG event capture validated
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

## NeuralDBG improvement — WHAT WE BUILT

### Test: `tests/unit/test_varlen_nan_detection.py`

**Problem solved**: BUG-002 requires CUDA hardware to reproduce. We can't run it on CPU.

**Solution**: Gradient injection test that simulates all 4 NaN/Inf patterns on CPU:
1. **NaN at tail** (padding positions) — the exact bug pattern
2. **NaN scattered** — different corruption patterns
3. **All NaN** — severe case
4. **Inf gradient** — overflow case

**How it works**:
1. Compute correct gradient on CPU (ground truth)
2. Inject NaN/Inf into the gradient tensor (simulating varlen_attn bug)
3. Verify NeuralDBG captures `gradient_health_transition` event
4. If CUDA available, run the same test with real varlen_attn

**Result**:
```
[DETECTED] Pattern 'nan_tail': NaN
  Events: 2 total, 1 anomaly-related
    gradient_health_transition: root
Result: PASS
```

**NeuralDBG code change**: None needed — the existing `gradient_health_transition` event type already handles NaN/Inf patterns. The test proves the detection works without CUDA.

### What this proves

NeuralDBG can detect NaN/Inf gradient corruption regardless of the source (varlen_attn, MHA, or any other module). The event capture system works for any gradient anomaly pattern.

## Relationship to other bugs

| Aspect | BUG-001 | BUG-002 | BUG-003 |
|--------|---------|---------|---------|
| Module | nn.MultiheadAttention | varlen_attn | MPS backend |
| Trigger | Fully masked row | Padding beyond cu_seqlens | Buffer pool reuse |
| Forward | Correct | Correct | Correct |
| Backward | NaN gradients | NaN gradients | Wrong magnitude gradients |
| Root cause | Composite module blind spot | Padding handling | MPS buffer corruption |
| NeuralDBG improvement | register_composite_hook() | NaN injection test | Gradient injection test |

## Deliverables checklist

- [x] BUG-002 tracking file (this file)
- [x] Test: `tests/unit/test_varlen_nan_detection.py` (NaN injection, no CUDA needed)
- [x] Detection confirmed: NeuralDBG captures `gradient_health_transition` on NaN gradients
- [ ] Reproduction script (`examples/repro_pytorch_176793.py`) — needs CUDA hardware
- [ ] Comment posted on pytorch/pytorch#176793 (CEO manual)
- [ ] Neural-Agent rule: "when NaN in attention gradient, check padding vs cu_seqlens alignment"

## Mom Test R2

- Test included with diagnostic output. No claim of fixing the upstream bug.
- Detection capability proven via gradient injection (hardware-independent approach).

## R64 Negative Mom Test

- What we don't detect: the specific varlen_attn C++ kernel bug
- What we DO detect: the consequence (NaN/Inf gradients) regardless of source module
