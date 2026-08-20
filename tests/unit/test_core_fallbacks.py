"""Regression tests for standalone Core behavior without neuraldbg-engine."""

import warnings
from pathlib import Path

import torch
import torch.nn as nn

import neuraldbg as neuraldbg_module
from neuraldbg import DataHealth, EventType, GradientHealth, NeuralDbg


def test_core_without_engine_detects_and_offloads_data_anomaly(monkeypatch, tmp_path):
    """Core fallback should not rely on the proprietary engine for NaN safety."""
    monkeypatch.setattr(neuraldbg_module, "_HAS_ENGINE", False)
    monkeypatch.setattr(neuraldbg_module, "CausalEngine", None)

    dbg = NeuralDbg(nn.Linear(3, 1))
    dbg.disk_cache.cache_dir = tmp_path
    dbg.step = 7

    dbg._check_data_anomaly(torch.tensor([[1.0, float("nan"), 3.0]]), "layer1")

    anomaly_events = [
        event for event in dbg.events if event.event_type == EventType.DATA_ANOMALY
    ]
    assert len(anomaly_events) == 1
    event = anomaly_events[0]
    assert event.from_state == DataHealth.NORMAL.value
    assert event.to_state == DataHealth.NAN_DETECTED.value
    assert event.metadata["nan_count"] == 1
    assert Path(event.metadata["tensor_cache_path"]).exists()


def test_core_without_engine_classifies_gradient_health(monkeypatch):
    """Gradient checks need heuristic fallbacks when neuraldbg-engine is absent."""
    monkeypatch.setattr(neuraldbg_module, "_HAS_ENGINE", False)
    monkeypatch.setattr(neuraldbg_module, "CausalEngine", None)

    dbg = NeuralDbg(
        nn.Linear(3, 1),
        threshold_vanishing=1e-4,
        threshold_exploding=10.0,
    )

    assert dbg._classify_gradient_health(1.0) == GradientHealth.HEALTHY
    assert dbg._classify_gradient_health(1e-8) == GradientHealth.VANISHING
    assert dbg._classify_gradient_health(100.0) == GradientHealth.EXPLODING


def test_core_without_engine_does_not_mislabel_missing_engine_as_inplace(monkeypatch):
    """Missing engine must not be swallowed as an inplace-operation warning."""
    monkeypatch.setattr(neuraldbg_module, "_HAS_ENGINE", False)
    monkeypatch.setattr(neuraldbg_module, "CausalEngine", None)

    model = nn.Linear(2, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with NeuralDbg(model) as dbg:
            model(torch.randn(1, 2)).sum().backward()

    # Healthy single step produces 0 events with debounce ≥2 (expected)
    assert not any("likely inplace operation" in str(item.message) for item in caught)
