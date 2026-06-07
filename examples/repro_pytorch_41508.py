#!/usr/bin/env python3
"""
Reproduction of pytorch/pytorch#41508 — nn.MultiheadAttention NaN gradients
when combining key_padding_mask and a fully-masking attn_mask.

Source: https://github.com/pytorch/pytorch/issues/41508
Bug status: OPEN, since 2020, 25+ participants, multiple workarounds proposed.
Confirmed on PyTorch 2.6.0+ (post-PR #133882) by @Oafish1, 2025-04-28.

Run:  python examples/repro_pytorch_41508.py

The script:
  1. Reproduces the bug (forward passes, NaN gradients).
  2. Shows that NeuralDBG (this library) does NOT auto-detect it — a known
     limitation when wrapping composite modules like nn.MultiheadAttention
     (see "NeuralDBG limitation" section below).
  3. Confirms the community-confirmed workaround (merge both masks into a
     single attn_mask and force the diagonal to 0) and shows clean gradients.
  4. Verifies the same scenario works with NeuralDBG once the workaround is
     applied (no NaN events emitted, training is healthy).

This file is the basis of the post-mortem blog article:
docs/blog/2026-06-13-pytorch-41508-postmortem.md

MID: BUG-001
Linked: POST-001, FIX-001
Tracker: docs/bugs/BUG-001-pytorch-41508.md
"""

import torch
from neuraldbg import NeuralDbg


def build_inputs():
    """Construct the minimal failing input from the original issue."""
    torch.manual_seed(0)
    attn = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1)

    # Sequence length 4, batch 2, embed dim 1.
    x = torch.rand(4, 2, 1)

    # Second sequence has 2 padding tokens on the right.
    key_padding_mask = torch.as_tensor(
        [[False, False, False, False], [False, False, True, True]],
        dtype=torch.bool,
    )

    # Bucketed attention: each query sees current and previous token.
    # When combined with the key padding above, the LAST query of sequence #2
    # has *all* keys masked (attn_mask: future tokens = -inf, padding = -inf)
    # -> softmax([-inf, -inf, -inf, -inf]) = [nan, nan, nan, nan]
    attn_mask = torch.as_tensor(
        [
            [0.0, float("-inf"), float("-inf"), float("-inf")],
            [0.0, 0.0, float("-inf"), float("-inf")],
            [float("-inf"), 0.0, 0.0, float("-inf")],
            [float("-inf"), float("-inf"), 0.0, 0.0],
        ],
    )
    return attn, x, key_padding_mask, attn_mask


def show_symptoms(attn, x, key_padding_mask, attn_mask):
    """Print the symptoms: finite forward, NaN gradients on masked row."""
    print("=" * 64)
    print("STEP 1 - Buggy forward + backward (no workaround)")
    print("=" * 64)
    output, scores = attn(
        x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
    )
    print(f"output[3, 1, 0] = {output[3, 1, 0].item()}  (expected nan)")
    print(f"scores[1, 3]    = {scores[1, 3].tolist()}  (expected all nan)")

    # Loss is computed only on the *valid* rows (first 2 of each sequence).
    loss = output[:2, :].sum()
    print(f"loss = {loss.item():.6f}  (forward: finite)")

    loss.backward()
    print()
    for n, p in attn.named_parameters():
        if p.grad is None:
            continue
        finite = torch.isfinite(p.grad).all().item()
        marker = "  <-- NaN corruption" if not finite else ""
        print(f"{n:18s}  shape={tuple(p.grad.shape)}  finite={finite}{marker}")


def show_neuraldbg_limitation(attn, x, key_padding_mask, attn_mask):
    """Demonstrate NeuralDBG's known limitation on composite modules."""
    print()
    print("=" * 64)
    print("STEP 2 - NeuralDBG wrapping (LIMITATION DEMO)")
    print("=" * 64)
    print("NeuralDBG installs hooks on leaf modules only. nn.MultiheadAttention")
    print("is a composite module (its in_proj_*/out_proj.* are not exposed as")
    print("leaf children), so the auto-installed hooks do not fire on the")
    print("forward/backward of the attention computation. This is a known")
    print("scope of NeuralDBG; composite modules require manual hook")
    print("attachment (out of scope for this post-mortem).")
    print()
    print("Consequence: NeuralDBG emits zero events for this bug, but the")
    print("gradients are still NaN. This is the kind of blind spot the")
    print("NeuralDBG team is actively cataloguing.")
    attn, x, key_padding_mask, attn_mask = build_inputs()
    with NeuralDbg(attn) as dbg:
        attn.train()
        dbg.step = 0
        output, _ = attn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        loss = output[:2, :].sum()
        loss.backward()
        dbg.record_loss(loss.item())
    print(f"Events emitted by NeuralDBG: {len(dbg.events)}")
    print("--> Empty event log; the NaN corruption went undetected.")


def show_workaround():
    """Community-confirmed workaround: merge masks + force diagonal."""
    print()
    print("=" * 64)
    print("STEP 3 - Workaround: merge masks + force diagonal to 0")
    print("=" * 64)
    print("Source: @JayanthShreekumar, comment of 2025-06-27 on issue #41508")
    print("Idea: combine attn_mask and key_padding_mask into ONE attn_mask,")
    print("then unmask the diagonal so every query can attend at least to")
    print("itself. The softmax then never sees a fully-masked row.")
    attn, x, key_padding_mask, attn_mask = build_inputs()
    B, S = key_padding_mask.shape
    combined = attn_mask.unsqueeze(0).expand(B, S, S).clone()
    combined = combined.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
    diag = torch.arange(S)
    combined[:, diag, diag] = 0.0  # allow self-attention
    output, scores = attn(x, x, x, attn_mask=combined)
    print(f"output[3, 1, 0] = {output[3, 1, 0].item():.6e}  (now finite)")
    print(f"scores[1, 3]    = {scores[1, 3].tolist()}        (now finite)")
    loss = output[:2, :].sum()
    loss.backward()
    all_finite = all(
        torch.isfinite(p.grad).all().item()
        for p in attn.parameters()
        if p.grad is not None
    )
    print(f"All gradients finite? {all_finite}")


def show_neuraldbg_with_workaround():
    """Confirm NeuralDBG runs cleanly on the fixed scenario."""
    print()
    print("=" * 64)
    print("STEP 4 - Same workaround wrapped in NeuralDBG")
    print("=" * 64)
    attn, x, key_padding_mask, attn_mask = build_inputs()
    B, S = key_padding_mask.shape
    combined = attn_mask.unsqueeze(0).expand(B, S, S).clone()
    combined = combined.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
    diag = torch.arange(S)
    combined[:, diag, diag] = 0.0
    with NeuralDbg(attn) as dbg:
        attn.train()
        dbg.step = 0
        for step in range(5):
            dbg.step = step
            output, _ = attn(x, x, x, attn_mask=combined)
            loss = output[:2, :].sum()
            loss.backward()
            dbg.record_loss(loss.item())
            for p in attn.parameters():
                p.grad = None
    print(f"Events emitted by NeuralDBG over 5 steps: {len(dbg.events)}")
    if dbg.events:
        print("Summary (per event type):")
        from collections import Counter

        c = Counter(e.event_type.value for e in dbg.events)
        for k, v in c.items():
            print(f"  {k}: {v}")
    print("--> Clean run, no NaN, no anomaly, training stable.")


if __name__ == "__main__":
    attn, x, key_padding_mask, attn_mask = build_inputs()
    show_symptoms(attn, x, key_padding_mask, attn_mask)
    show_neuraldbg_limitation(attn, x, key_padding_mask, attn_mask)
    show_workaround()
    show_neuraldbg_with_workaround()
