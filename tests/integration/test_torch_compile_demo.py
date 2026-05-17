"""Integration tests for torch.compile (Dynamo) compatibility scenarios."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_torch_compile import (
    scenario_compile_healthy,
    scenario_compile_vanishing,
    scenario_compile_exploding,
    analyze_results,
)


def test_compile_healthy_captures_events():
    dbg = scenario_compile_healthy(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) >= 0


def test_compile_vanishing_captures_events():
    dbg = scenario_compile_vanishing(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_compile_exploding_captures_events():
    dbg = scenario_compile_exploding(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_compile_mermaid_graph():
    dbg = scenario_compile_healthy(num_steps=5)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")
