# BUG-001 — PyTorch #41508 NaN gradients in nn.MultiheadAttention

> **MID**: BUG-001
> **Linked**: POST-001 (postmortem), FIX-001 (NeuralDBG composite hooks)
> **Status**: Workaround confirmed — NeuralDBG blind spot acknowledged — fix in progress (v1.3.2)
> **Date opened**: 2026-06-13
> **Owner**: LambdaSection

## Source

- Upstream issue: https://github.com/pytorch/pytorch/issues/41508
- Title: *"nn.MultiheadAttention causes gradients to become NaN under some use cases"*
- Status upstream: OPEN, since July 2020, 25+ participants
- Confirmed on PyTorch 2.6.0 (post-PR #133882) by @Oafish1, 2025-04-28

## Trigger conditions

A row is fully masked by the combination of `attn_mask` and `key_padding_mask` (i.e. every column in that row is `-inf`). Backward pass through that row produces NaN gradients in `in_proj_weight` and `in_proj_bias`.

Minimal repro:

```python
torch.manual_seed(0)
attn = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1)
x = torch.rand(4, 2, 1)
key_padding_mask = torch.as_tensor(
    [[False, False, False, False],
     [False, False, True,  True]], dtype=torch.bool)
attn_mask = torch.as_tensor(
    [[0., float('-inf'), float('-inf'), float('-inf')],
     [0., 0., float('-inf'), float('-inf')],
     [float('-inf'), 0., 0., float('-inf')],
     [float('-inf'), float('-inf'), 0., 0.]])
output, scores = attn(x, x, x,
                      key_padding_mask=key_padding_mask,
                      attn_mask=attn_mask)
loss = output[:2, :].sum()
loss.backward()
# -> in_proj_weight.grad contains NaN
```

## Reproduction script

`examples/repro_pytorch_41508.py` (178 lines).

Runs 4 stages:
1. Reproduce the bug (NaN gradients emitted by PyTorch).
2. Show NeuralDBG **does not** auto-detect it (composite-module blind spot).
3. Apply the community workaround (merge masks, force diagonal to 0).
4. Re-run with NeuralDBG: no NaN events, training healthy.

## NeuralDBG blind spot (the lesson)

`nn.MultiheadAttention` is a **composite** module: it has no leaf children (its parameters live on internal `Linear` submodules, but the autograd flow goes through a custom C++ kernel that bypasses the leaf hooks). NeuralDBG's auto leaf-only hook installer was therefore silent on this bug.

Documented in detail in POST-001 ("What we missed and why").

## Workaround (community-confirmed)

Merge `key_padding_mask` into `attn_mask`, then force the diagonal of `attn_mask` to 0 so no row is ever fully masked. Clean gradients, forward unchanged.

## NeuralDBG fix delivered (FIX-001)

- New public API: `NeuralDbg.register_composite_hook(module)`
- New warning at `__enter__` when the wrapped model exposes zero leaf modules
- New warning at `__exit__` when `step >= 3` and no `gradient_health_transition` event was ever captured
- Backward-compatible: default behaviour unchanged for normal models

Lands in **v1.3.2**.

## Linked NeuralAgent improvement (planned)

- Agent remediation rule: when a fully-masked row pattern is detected in attention masks, suggest the merged-mask workaround
- Tied to FIXME in `Neural-Agent/`: hook the postmortem example as a unit test for the remediation runner

## Deliverables checklist

- [x] Reproduction script (`examples/repro_pytorch_41508.py`)
- [x] Postmortem blog (`docs/blog/2026-06-13-pytorch-41508-postmortem.md` + `.html`)
- [x] BUG-001 tracking file (this file)
- [x] Workaround documented and verified
- [ ] FIX-001 merged (PR open, v1.3.2 release)
- [ ] Comment posted on pytorch/pytorch#41508 with link to postmortem
- [ ] NeuralAgent remediation rule added
- [ ] Tweet/thread with diagnostic capture

## Sign-off

- Mom Test R2: reproduction script + diagnostic log included. No claim of fixing the upstream bug — only the workaround and the NeuralDBG blind spot are owned.
- R64 Negative Mom Test: what we *don't* detect is also documented (MHA without explicit `register_composite_hook`).
