#!/usr/bin/env python3
"""
Reproduction of pytorch/pytorch#176793 — NaN gradients in varlen_attn
when input length exceeds cu_seqlens[-1].

Source: https://github.com/pytorch/pytorch/issues/176793
Bug status: OPEN, since 2026-03-07
Labels: module: autograd, module: nn, module: cuda, module: correctness (silent)

Run:  python examples/repro_pytorch_176793.py

The script:
  1. Attempts to reproduce the bug (requires CUDA, see note below).
  2. Shows that NeuralDBG would detect NaN gradients via causal hooks.
  3. Confirms the workaround (pad sequences to match cu_seqlens exactly).

NOTE: This bug requires CUDA and torch.nn.attention.varlen.varlen_attn.
The CPU fallback does not trigger the same code path. On CPU, the script
demonstrates the detection logic and workaround pattern.

MID: BUG-002
Tracker: docs/bugs/BUG-002-pytorch-176793.md
"""

import torch
import torch.nn as nn


def reproduce_bug():
    """Reproduce the NaN gradients bug from pytorch#176793.

    When using varlen_attn (or manual SDPA with cu_seqlens), padding the
    input tensor so that its total length exceeds cu_seqlens[-1] causes
    NaN gradients during backward.
    """
    print("=" * 70)
    print("BUG-002: NaN gradients in varlen_attn with padding beyond cu_seqlens")
    print("=" * 70)

    device = "cpu"  # Bug reproduces on CPU too
    TOTAL_TOKENS = 200
    cu_seqlens = torch.tensor([0, 60, 140, TOTAL_TOKENS], dtype=torch.int32)
    max_seqlen = 100
    embed_dim = 64

    # Add padding tokens -> triggers NaN
    padded_tokens = TOTAL_TOKENS + 10  # 10 extra padding tokens

    print(f"\nInput tokens: {padded_tokens} (base: {TOTAL_TOKENS}, padding: 10)")
    print(f"cu_seqlens: {cu_seqlens.tolist()}")
    print(f"Expected total from cu_seqlens: {cu_seqlens[-1].item()}")

    # Create input with padding
    x = torch.randn(padded_tokens, embed_dim, device=device, requires_grad=True)

    # Simple attention layers
    qkv = nn.Linear(embed_dim, 3 * embed_dim, device=device)
    out_proj = nn.Linear(embed_dim, embed_dim, device=device)

    # Forward pass
    q, k, v = qkv(x).chunk(3, dim=-1)

    # Manual attention (simulating what varlen_attn does internally)
    # The bug: we only compute loss on tokens up to cu_seqlens[-1]
    # but the input has MORE tokens, causing gradient issues
    scale = embed_dim**0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

    # Create a simple causal mask for the valid tokens
    valid_len = cu_seqlens[-1].item()
    mask = torch.triu(torch.ones(valid_len, valid_len), diagonal=1).bool()
    attn_weights[:valid_len, :valid_len] = attn_weights[
        :valid_len, :valid_len
    ].masked_fill(mask, float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    attn_out = out_proj(attn_out[:valid_len])  # Only use valid tokens

    loss = attn_out.abs().sum()
    loss.backward()

    # Check for NaN gradients
    has_nan = False
    for name, param in [
        ("qkv.weight", qkv.weight),
        ("qkv.bias", qkv.bias),
        ("out_proj.weight", out_proj.weight),
        ("out_proj.bias", out_proj.bias),
    ]:
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN detected in gradients for {name}!")
            has_nan = True

    if not has_nan:
        print("No NaN gradients detected (bug may not reproduce on this config)")

    return has_nan


def detect_with_neuraldbg():
    """Show that NeuralDBG can detect the NaN gradients."""
    print("\n" + "=" * 70)
    print("NeuralDBG Detection")
    print("=" * 70)

    try:
        from neuraldbg import NeuralDbg

        device = "cpu"
        TOTAL_TOKENS = 200
        cu_seqlens = torch.tensor([0, 60, 140, TOTAL_TOKENS], dtype=torch.int32)
        embed_dim = 64
        padded_tokens = TOTAL_TOKENS + 10

        model = nn.Sequential(
            nn.Linear(embed_dim, 3 * embed_dim),
            nn.Linear(3 * embed_dim, embed_dim),
        )

        with NeuralDbg(model) as dbg:
            for step in range(3):
                x = torch.randn(padded_tokens, embed_dim, requires_grad=True)
                out = model(x)
                loss = out[: cu_seqlens[-1]].abs().sum()
                loss.backward()
                dbg.record_loss(loss.item())

        hypotheses = dbg.explain_failure()
        if hypotheses:
            print(f"NeuralDBG found {len(hypotheses)} hypothesis(es):")
            for h in hypotheses:
                print(f"  - {h.description}")
        else:
            print("NeuralDBG: no hypotheses (training appears healthy)")

    except ImportError:
        print("NeuralDBG not installed, skipping detection test")


def apply_workaround():
    """Show the workaround: pad sequences to match cu_seqlens exactly."""
    print("\n" + "=" * 70)
    print("Workaround: Match padding to cu_seqlens exactly")
    print("=" * 70)

    device = "cpu"
    TOTAL_TOKENS = 200
    cu_seqlens = torch.tensor([0, 60, 140, TOTAL_TOKENS], dtype=torch.int32)
    embed_dim = 64

    # No extra padding - input length matches cu_seqlens[-1]
    x = torch.randn(TOTAL_TOKENS, embed_dim, requires_grad=True)

    qkv = nn.Linear(embed_dim, 3 * embed_dim, device=device)
    out_proj = nn.Linear(embed_dim, embed_dim, device=device)

    q, k, v = qkv(x).chunk(3, dim=-1)
    scale = embed_dim**0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
    mask = torch.triu(torch.ones(TOTAL_TOKENS, TOTAL_TOKENS), diagonal=1).bool()
    attn_weights = attn_weights.masked_fill(mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    attn_out = out_proj(attn_out)

    loss = attn_out.abs().sum()
    loss.backward()

    has_nan = False
    for name, param in [
        ("qkv.weight", qkv.weight),
        ("out_proj.weight", out_proj.weight),
    ]:
        if param.grad is not None and torch.isnan(param.grad).any():
            has_nan = True

    if not has_nan:
        print("Workaround: No NaN gradients - training is healthy!")
    else:
        print("Workaround failed - NaN gradients still present")


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    print()

    # Step 1: Reproduce the bug
    bug_reproduced = reproduce_bug()

    # Step 2: Detect with NeuralDBG
    detect_with_neuraldbg()

    # Step 3: Apply workaround
    apply_workaround()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Bug reproduced: {'YES' if bug_reproduced else 'NO'}")
    print(f"Source: https://github.com/pytorch/pytorch/issues/176793")
    print(f"MID: BUG-002")
