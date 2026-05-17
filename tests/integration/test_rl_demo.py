"""Integration tests for RL (PPO-style) failure scenarios demo."""

import pytest

torch = pytest.importorskip("torch")

from examples.demo_rl_failures import (
    scenario_policy_collapse,
    scenario_value_explosion,
    scenario_reward_hacking,
    analyze_results,
)


def test_policy_collapse_captures_events():
    dbg = scenario_policy_collapse(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_value_explosion_captures_events():
    dbg = scenario_value_explosion(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_reward_hacking_captures_events():
    dbg = scenario_reward_hacking(num_steps=5)
    results = analyze_results(dbg)
    assert len(results["events"]) > 0


def test_rl_mermaid_graph():
    dbg = scenario_policy_collapse(num_steps=5)
    results = analyze_results(dbg)
    assert results["mermaid"].startswith("graph TD")


def test_rl_couplings_deduplicated():
    dbg = scenario_value_explosion(num_steps=8)
    results = analyze_results(dbg)
    couplings = results["couplings"]
    pair_set = set()
    for c in couplings:
        pair = (c["trigger"], c["consequence"])
        assert pair not in pair_set
        pair_set.add(pair)
