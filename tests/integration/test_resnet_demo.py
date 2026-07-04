"""NeuralDBG. Copyright (c) 2026 NeuralDBG."""
"""Integration tests for ResNet-18 failure scenarios demo."""

import pytest

pytest.importorskip("torchvision")

from examples.demo_resnet_failures import (
    scenario_vanishing_gradients,
    scenario_exploding_gradients,
    scenario_data_anomaly,
    analyze_results,
)


def test_vanishing_gradients_produces_hypotheses():
    dbg = scenario_vanishing_gradients(num_steps=10)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0
    assert len(results["hypotheses"]) > 0
    for h in results["hypotheses"]:
        assert 0.0 <= h.confidence <= 1.0
    # Sanity: vanishing hypotheses reference vanishing state
    hyps = dbg.explain_failure("vanishing_gradients")
    if hyps:
        assert any("vanishing" in h.description.lower() for h in hyps)


def test_exploding_gradients_captures_events():
    dbg = scenario_exploding_gradients(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_data_anomaly_detects_nan():
    dbg = scenario_data_anomaly(num_steps=8)
    results = analyze_results(dbg)
    assert len(results["data_hypotheses"]) > 0
    nan_hyps = [h for h in results["data_hypotheses"] if "nan" in h.description.lower()]
    assert len(nan_hyps) > 0


def test_mermaid_graph_generated():
    dbg = scenario_vanishing_gradients(num_steps=10)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")
    assert "E_" in results["mermaid"]  # UUID-based event IDs


def test_couplings_detected():
    dbg = scenario_vanishing_gradients(num_steps=10)
    results = analyze_results(dbg)
    couplings = results["couplings"]
    pair_set = set()
    for c in couplings:
        pair = (c["trigger"], c["consequence"])
        assert pair not in pair_set
        pair_set.add(pair)
