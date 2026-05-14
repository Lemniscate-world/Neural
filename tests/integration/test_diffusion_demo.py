"""Integration tests for DDPM diffusion failure scenarios demo."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_diffusion_failures import (
    scenario_unet_nan,
    scenario_unet_exploding,
    scenario_unet_collapse,
    analyze_results,
)


def test_unet_nan_captures_events():
    dbg = scenario_unet_nan(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_unet_exploding_captures_events():
    dbg = scenario_unet_exploding(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_unet_collapse_captures_events():
    dbg = scenario_unet_collapse(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_diffusion_mermaid_graph():
    dbg = scenario_unet_nan(num_steps=5)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")
