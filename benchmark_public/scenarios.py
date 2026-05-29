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


def _noop(_model, _step):
    pass


def _vanishing_injector(model, step):
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
            bug_layer="4",
            bug_step=2,
            expected_hypothesis_substring="vanish",
            expected_bug_layer="4",
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
            bug_layer="0",
            bug_step=3,
            expected_hypothesis_substring="explod",
            expected_bug_layer="0",
        ),
    ),
]
