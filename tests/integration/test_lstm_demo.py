"""Integration tests for LSTM / Time-Series failure scenarios demo."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_lstm_failures import (
    scenario_vanishing_recurrent,
    scenario_exploding_recurrent,
    scenario_deep_lstm,
    analyze_results,
)


def test_vanishing_recurrent_captures_events():
    dbg = scenario_vanishing_recurrent(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_exploding_recurrent_captures_events():
    dbg = scenario_exploding_recurrent(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_deep_lstm_captures_events():
    dbg = scenario_deep_lstm(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_lstm_mermaid_graph():
    dbg = scenario_vanishing_recurrent(num_steps=5)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")


def test_lstm_couplings_deduplicated():
    dbg = scenario_exploding_recurrent(num_steps=8)
    results = analyze_results(dbg)
    couplings = results["couplings"]
    pair_set = set()
    for c in couplings:
        pair = (c["trigger"], c["consequence"])
        assert pair not in pair_set
        pair_set.add(pair)
