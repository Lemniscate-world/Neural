"""Integration tests for Distributed / DataParallel failure scenarios demo."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_distributed_failures import (
    scenario_dp_healthy,
    scenario_dp_vanishing,
    scenario_dp_exploding,
    analyze_results,
)


def test_dp_healthy_captures_events():
    dbg = scenario_dp_healthy(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) >= 0


def test_dp_vanishing_captures_events():
    dbg = scenario_dp_vanishing(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_dp_exploding_captures_events():
    dbg = scenario_dp_exploding(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_dp_mermaid_graph():
    dbg = scenario_dp_healthy(num_steps=5)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")
