"""Integration tests for LoRA fine-tuning failure scenarios demo."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_lora_finetune import (
    scenario_nan_lora,
    scenario_exploding_lora,
    scenario_forgetting_lora,
    analyze_results,
)


def test_nan_lora_captures_events():
    dbg = scenario_nan_lora(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_exploding_lora_captures_events():
    dbg = scenario_exploding_lora(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_forgetting_lora_captures_events():
    dbg = scenario_forgetting_lora(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0
