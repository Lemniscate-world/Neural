"""Synthetic failure scenarios with ground truth for public benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

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
