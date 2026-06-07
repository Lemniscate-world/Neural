---
title: "Post-mortem: PyTorch issue #41508 — NaN gradients in nn.MultiheadAttention"
date: 2026-06-13
author: LambdaSection
tags: [pytorch, debugging, attention, postmortem, real-bug, BUG-001, POST-001]
description: "A real PyTorch bug (open since 2020, 25+ participants) dissected: when does MHA produce NaN gradients, why can't NeuralDBG auto-detect it, and what is the community-confirmed workaround."
mid: POST-001
bug: BUG-001
fix: FIX-001
---

# Post-mortem: PyTorch issue #41508 — NaN gradients in `nn.MultiheadAttention`

> **TL;DR.** A bug in `nn.MultiheadAttention`, open since 2020 with 25+ participants, produces NaN gradients when `attn_mask` and `key_padding_mask` together leave a row fully masked. We reproduce it, document NeuralDBG's blind spot on composite modules, and confirm the community workaround. Honest about what we can and cannot detect.

---

## The bug

[`pytorch/pytorch#41508`](https://github.com/pytorch/pytorch/issues/41508) — *"nn.MultiheadAttention causes gradients to become NaN under some use cases"*. Open since July 2020, still reproducible on PyTorch 2.6.0 as of April 2025.

**Repro** (from the issue, slightly simplified):

```python
import torch

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
loss = output[:2, :].sum()  # only valid rows
loss.backward()
```

Running this on PyTorch 2.6.0 produces:

```
output[3, 1, 0] = nan           # the fully-masked row
scores[1, 3]    = [nan, nan, nan, nan]
loss            = 0.008393     # forward is finite

in_proj_weight   shape=(3, 1)   finite=False   <-- NaN corruption
in_proj_bias     shape=(3,)     finite=False   <-- NaN corruption
out_proj.weight  shape=(1, 1)   finite=False   <-- NaN corruption
out_proj.bias    shape=(1,)     finite=True    # only because not used by masked row
```

The forward pass produces a finite loss (because we sliced `output[:2]` to skip the NaN row). The backward pass corrupts all learnable parameters. **One masked row is enough to break the whole layer.**

## Why this happens

The PyTorch team explained it on the issue: MHA implements masking by passing a large negative bias (`-inf`) to the attention scores before softmax. A row that is fully masked produces `softmax([-inf, -inf, -inf, -inf]) = [nan, nan, nan, nan]`. The NaN scores are then used during backward, propagating corruption to all weight gradients.

Mathematically this is the *correct* behavior of softmax. The issue is that the *output* of MHA on a fully-masked row should arguably be `0` or undefined — but in any case, it should not contaminate gradients on rows the user is actually using for the loss.

The PR that attempted to fix this ([#133882](https://github.com/pytorch/pytorch/pull/133882), merged via co-dev in August 2024) added a private `_safe_softmax` op that returns 0 for fully-masked rows. As of April 2025, users on PyTorch 2.6.0 still report the bug is reproducible.

## Where NeuralDBG fits (and where it doesn't)

NeuralDBG auto-installs forward/backward hooks on **leaf modules** of a model. `nn.MultiheadAttention` is a *composite* module — it owns `in_proj_weight`, `in_proj_bias`, `out_proj.weight`, `out_proj.bias` as `nn.Parameter`s, but it has no leaf submodules. When you wrap a bare MHA in `NeuralDbg(attn)`, the auto-hook installer skips it, and the engine captures **no events** for the bug — even though the gradients are NaN.

We tried it. The output:

```python
attn = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1)
with NeuralDbg(attn) as dbg:
    # ... same forward + backward as above ...
    pass

print(len(dbg.events))  # 1 (an activation baseline, not a NaN detection)
```

This is a real, documented limitation. NeuralDBG is optimised for sequential architectures (Linear → Activation → …) and convolutional / recurrent modules that decompose into leaf ops. Composite transformer blocks — MHA, the various `*Norm` layers wrapping leaf ops inside, fused QKV projections — fall outside the auto-instrumentation surface.

**This is not the end of the story.** Three things are possible:

1. **Manual hook installation** on `attn` via `register_full_backward_hook` works (we verified the API path is reachable). It is just not part of the public `NeuralDbg` API yet.
2. **Wrapping the inner matmuls** (extracting `W_q`, `W_k`, `W_v` from `in_proj_weight` and instantiating a `MultiheadAttention` with `in_proj_weight=None` + explicit `W_q`/`W_k`/`W_v` modules) would make the components visible as leaf modules. Not always feasible.
3. **A future version of NeuralDBG** could expose a `composite=True` flag that installs hooks at the module level for known patterns. Tracked internally.

We are not going to claim NeuralDBG auto-detects this bug today. **It doesn't.** But we are going to document the limitation and the workaround.

## The community workaround

The cleanest workaround, proposed by [`@JayanthShreekumar`](https://github.com/JayanthShreekumar) on June 27 2025, is to:

1. **Merge** `attn_mask` and `key_padding_mask` into a single `attn_mask`.
2. **Force the diagonal to 0** so every query attends to at least itself.
3. **Pass the result to `attn_mask=`**, leaving `key_padding_mask=None`.

```python
B, S = key_padding_mask.shape
combined = attn_mask.unsqueeze(0).expand(B, S, S).clone()
combined = combined.masked_fill(key_padding_mask.unsqueeze(1), float('-inf'))
diag = torch.arange(S)
combined[:, diag, diag] = 0.0  # unmask self-attention

output, scores = attn(x, x, x, attn_mask=combined)
```

Result:

```
output[3, 1, 0] = 7.884680e-05   # finite
scores[1, 3]    = [0.0, 0.0, 0.0, 1.0]  # clean self-attention
All gradients finite? True
```

The fully-masked row no longer exists: every query attends to at least itself, so the softmax denominator is always positive. Mathematically sound, no NaN possible, no change to the gradient flow on the unmasked rows.

The previous variant proposed in the issue (add `torch.eye` to the mask) is a special case of the same idea.

## The clean run with NeuralDBG

After applying the workaround, the same scenario wrapped in NeuralDBG over 5 steps produces **two events**: a single activation regime shift baseline at step 0 (the attention output is small but non-anomalous), and an activation baseline at step 1. **No `gradient_health_transition`, no `data_anomaly`, no `optimizer_instability`.** That's exactly what a healthy run should look like in the event log.

```python
with NeuralDbg(attn) as dbg:
    for step in range(5):
        dbg.step = step
        output, _ = attn(x, x, x, attn_mask=combined)
        loss = output[:2, :].sum()
        loss.backward()
        dbg.record_loss(loss.item())
```

```
Events emitted by NeuralDBG over 5 steps: 2
Summary (per event type):
  activation_regime_shift: 2
--> Clean run, no NaN, no anomaly, training stable.
```

## What we learned (and what's next for NeuralDBG)

This is the third category of bug we have catalogued in our `docs/bug_hunt_charter.md`:

| Category | Detection status | Mitigation |
|---|---|---|
| Standard sequential model (Linear/Tanh) | ✅ Auto-detected | Activation regime shift event |
| Conv / residual block (ResNet) | ✅ Auto-detected | Gradient health transition + first-occurrence ranking |
| **Composite transformer block (MHA, custom fused layers)** | ❌ **Auto-blind spot** | **Manual hook or workaround** |

The action items that fall out of this post-mortem:

1. **Expose `register_composite_hook` in NeuralDBG's public API** for users to instrument known composite modules (tracked internally).
2. **Add a "silent loss" detection** that flags when `loss.backward()` does not produce gradient events despite being called — the failure mode of this bug, but at a higher level. (Currently we rely on activation hooks; if none fire, we are silent.)
3. **Update the standalone fallback warnings** to mention composite modules explicitly when they are the root of the model.
4. **Document this case in the user guide** with the workaround, so users who hit it find the recipe.

None of these change NeuralDBG's core engine. They are integration improvements driven by a real failure mode we found in the wild.

## Try the repro

```bash
pip install neuraldbg
python examples/repro_pytorch_41508.py
```

The script runs in about a second on CPU. It shows all four steps: the bug, the NeuralDBG limitation, the workaround, and the clean run.

## Credits

- **[@wgale](https://github.com/wgale)** — original bug report (PyTorch issue #41508, July 2020).
- **[@zhangguanheng66](https://github.com/zhangguanheng66)** — early explanation of the softmax-NaN root cause.
- **[@chenxie95](https://github.com/chenxie95)** — additional reproduction and the observation that `x.grad[:, 0, :]` was valid but `x.grad[:, 1, :]` was NaN.
- **[@JayanthShreekumar](https://github.com/JayanthShreekumar)** — the merged-mask + diagonal-unmask workaround (June 2025).
- **[@drisspg](https://github.com/drisspg)** and the PyTorch core team — the merged PR #133882 (`_safe_softmax`).

We stand on their shoulders. This post-mortem is a small contribution back to the same conversation.

---

**Related**

- [GitHub issue: pytorch/pytorch#41508](https://github.com/pytorch/pytorch/issues/41508)
- [PyTorch PR #133882 — _safe_softmax](https://github.com/pytorch/pytorch/pull/133882)
- [NeuralDBG bug-hunt charter](bug_hunt_charter.md)
- [Standalone library: github.com/LambdaSection/NeuralDBG](https://github.com/LambdaSection/NeuralDBG) (MIT)
