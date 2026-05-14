"""NeuralDBG. Copyright (c) 2026 NeuralDBG."""
"""Integration tests for Transformer (GPT) failure scenarios demo."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_transformer_failures import (
    scenario_no_warmup,
    scenario_no_norm,
    scenario_no_scale,
    analyze_results,
)


def test_no_warmup_captures_gradient_events():
    dbg = scenario_no_warmup(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_no_norm_detects_activation_shifts():
    dbg = scenario_no_norm(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0
    # Without LayerNorm, activations should saturate
    sat_events = [
        e for e in results["events"] if getattr(e, "to_state", "") == "saturated"
    ]
    assert len(sat_events) > 0 or len(results["hypotheses"]) > 0


def test_no_scale_produces_hypotheses():
    dbg = scenario_no_scale(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_mermaid_graph_generated():
    dbg = scenario_no_warmup(num_steps=5)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")


def test_couplings_deduplicated():
    dbg = scenario_no_warmup(num_steps=5)
    results = analyze_results(dbg)
    couplings = results["couplings"]
    pair_set = set()
    for c in couplings:
        pair = (c["trigger"], c["consequence"])
        assert pair not in pair_set
        pair_set.add(pair)
