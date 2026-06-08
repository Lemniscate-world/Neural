"""Synthetic failure scenarios with ground truth for public benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GroundTruth:
    bug_type: str
    bug_layer: str
    bug_step: int
    expected_hypothesis_substring: str
    expected_bug_layer: str


@dataclass
class Scenario:
    name: str
    model_builder: Callable[[], nn.Module]
    data_builder: Callable[[], Tuple[torch.Tensor, torch.Tensor]]
    num_steps: int
    bug_injector: Callable[[nn.Module, int], None]
    ground_truth: GroundTruth
    # Optional custom step loop. If None, the runner uses the default
    # classification loop (forward -> CE loss -> backward). The custom
    # loop is needed for non-classification scenarios (e.g. MHA which
    # takes attn_mask + key_padding_mask).
    step_fn: Optional[Callable[[Any, Any, Any, Any, int], None]] = None
    # Optional metadata: maps the scenario to its MIDs and source.
    mid: Optional[str] = None
    source: Optional[str] = None


def _deep_sigmoid_mlp():
    """Deep MLP with Sigmoid activations — gradients vanish naturally.

    8 Linear+Sigmoid blocks. The sigmoid derivative maxes at 0.25, so after
    6+ layers the gradient signal is reduced by ~0.25^6 ≈ 2.4e-4. This is a
    realistic architectural failure — no artificial injection needed.
    """
    layers = []
    in_features = 16
    hidden = 32
    for i in range(8):
        layers.append(nn.Linear(in_features if i == 0 else hidden, hidden))
        layers.append(nn.Sigmoid())
    layers.append(nn.Linear(hidden, 4))
    return nn.Sequential(*layers)


def _mlp():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.Tanh(),
        nn.Linear(20, 20),
        nn.Tanh(),
        nn.Linear(20, 2),
    )


def _data():
    return torch.randn(8, 10), torch.randint(0, 2, (8,))


def _deep_data():
    return torch.randn(8, 16), torch.randint(0, 4, (8,))


def _noop(_model, _step):
    pass


def _vanishing_injector(model, step):
    """Inject vanishing gradients by zeroing all parameters at step 2.

    This creates a clear, unambiguous transition from healthy to vanishing
    gradients across the entire network. While artificial, this is the most
    reliable way to test NeuralDBG's detection and localization of vanishing
    gradients — a real-world scenario that this mirrors is catastrophic
    weight corruption (e.g., NaN propagation, checkpoint corruption).
    """
    if step == 2:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(0.0)


def _exploding_injector(model, step):
    if step == 3:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(50.0)


PUBLIC_SCENARIOS = [
    Scenario(
        name="healthy_training",
        model_builder=lambda: nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        ),
        data_builder=_data,
        num_steps=15,
        bug_injector=_noop,
        ground_truth=GroundTruth(
            bug_type="none",
            bug_layer="",
            bug_step=-1,
            expected_hypothesis_substring="",
            expected_bug_layer="",
        ),
    ),
    Scenario(
        name="vanishing_gradients",
        model_builder=_mlp,
        data_builder=_data,
        num_steps=10,
        bug_injector=_vanishing_injector,
        ground_truth=GroundTruth(
            bug_type="vanishing_gradients",
            bug_layer="Tanh_3",
            bug_step=2,
            expected_hypothesis_substring="vanish",
            expected_bug_layer="Tanh_3",
        ),
    ),
    Scenario(
        name="exploding_gradients",
        model_builder=_mlp,
        data_builder=_data,
        num_steps=10,
        bug_injector=_exploding_injector,
        ground_truth=GroundTruth(
            bug_type="exploding_gradients",
            bug_layer="Linear_0",
            bug_step=3,
            expected_hypothesis_substring="explod",
            expected_bug_layer="Linear_0",
        ),
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# BUG-001 / pytorch/pytorch#41508 — NaN gradients in nn.MultiheadAttention
# when a row is fully masked by attn_mask + key_padding_mask combined.
#
# This is a REAL upstream bug (open since 2020, 25+ participants) and the
# reason FIX-001 (register_composite_hook) exists in NeuralDBG v1.3.2.
# Reproduction: examples/repro_pytorch_41508.py.
#
# Loss is finite (forward masks the NaN row out of the loss). Gradients
# on in_proj_weight / in_proj_bias / out_proj.weight are NaN. A naive
# loss logger (W&B-style) would NEVER see this — that's the whole point
# of putting it in the benchmark.
# ──────────────────────────────────────────────────────────────────────────────


def _mha_bug_model():
    return nn.MultiheadAttention(embed_dim=1, num_heads=1)


def _mha_bug_data():
    """Return (x, attn_mask, key_padding_mask) — the exact failing input."""
    x = torch.rand(4, 2, 1)
    kpm = torch.as_tensor(
        [[False, False, False, False], [False, False, True, True]],
        dtype=torch.bool,
    )
    am = torch.as_tensor(
        [
            [0.0, float("-inf"), float("-inf"), float("-inf")],
            [0.0, 0.0, float("-inf"), float("-inf")],
            [float("-inf"), 0.0, 0.0, float("-inf")],
            [float("-inf"), float("-inf"), 0.0, 0.0],
        ]
    )
    return x, am, kpm


def _mha_bug_step(model, dbg, x, am_kpm, step):
    """Custom step loop: feed MHA with attn_mask + key_padding_mask."""
    x_t, am, kpm = am_kpm[0], am_kpm[1], am_kpm[2]
    out, _ = model(x_t, x_t, x_t, attn_mask=am, key_padding_mask=kpm)
    loss = out[:2, :].sum()
    loss.backward()
    dbg.record_loss(loss.item())


PUBLIC_SCENARIOS.append(
    Scenario(
        name="mha_fully_masked_row_BUG-001",
        model_builder=_mha_bug_model,
        data_builder=_mha_bug_data,
        num_steps=3,
        bug_injector=_noop,
        step_fn=_mha_bug_step,
        mid="BUG-001",
        source="https://github.com/pytorch/pytorch/issues/41508",
        ground_truth=GroundTruth(
            bug_type="composite_blind_spot",
            bug_layer="root",
            bug_step=0,
            expected_hypothesis_substring="composite",
            expected_bug_layer="root",
        ),
    )
)


# ---------------------------------------------------------------------------
# NaN loss from direct NaN injection into layer output.
#
# This simulates a common real-world failure: a layer produces NaN values
# (e.g., from log(0), division by zero, or numerical instability) which
# propagate to the loss. External tools (W&B, MLflow, TensorBoard) can
# detect NaN loss but cannot localize which layer caused it.
#
# NeuralDBG hooks at parameter level and can identify the failing layer.
# ---------------------------------------------------------------------------


def _nan_loss_model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )


def _nan_loss_data():
    return torch.randn(8, 10), torch.randint(0, 2, (8,))


def _nan_loss_injector(model, step):
    """Inject NaN into the second Linear layer's output at step 3.

    This causes the loss to become NaN from step 3 onward. The injection
    happens on the output of Linear_2 (the middle layer), making it the
    root cause layer for localization.
    """
    if step == 3:
        original_forward = model[2].forward

        def nan_forward(x):
            out = original_forward(x)
            return out.fill_(float("nan"))

        model[2].forward = nan_forward


PUBLIC_SCENARIOS.append(
    Scenario(
        name="nan_loss_from_layer_injection",
        model_builder=_nan_loss_model,
        data_builder=_nan_loss_data,
        num_steps=10,
        bug_injector=_nan_loss_injector,
        ground_truth=GroundTruth(
            bug_type="nan_loss",
            bug_layer="Linear_2",
            bug_step=3,
            expected_hypothesis_substring="nan",
            expected_bug_layer="Linear_2",
        ),
    )
)
