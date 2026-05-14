"""Integration tests for GAN generator failure scenarios demo."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_gan_failures import (
    scenario_vanishing_generator,
    scenario_exploding_generator,
    scenario_generator_nan,
    analyze_results,
)


def test_vanishing_generator_captures_events():
    dbg = scenario_vanishing_generator(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_exploding_generator_captures_events():
    dbg = scenario_exploding_generator(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_generator_nan_detected():
    dbg = scenario_generator_nan(num_steps=8)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0
    nan_hyps = [h for h in results["data_hypotheses"] if "nan" in h.description.lower()]
    assert len(nan_hyps) > 0


def test_mermaid_graph_generated():
    dbg = scenario_exploding_generator(num_steps=5)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")
