"""Integration test for semantic explanations in the demo."""

import pytest

pytest.importorskip("torch")

from examples.demo_vanishing_gradients import (
    create_failing_model,
    create_problematic_data,
    train_with_monitoring,
    analyze_results,
)


def test_demo_produces_causal_hypotheses():
    """Verify the demo produces reasonable causal hypotheses for vanishing gradients."""
    model = create_failing_model()
    dataloader = create_problematic_data()
    
    dbg = train_with_monitoring(model, dataloader, num_steps=10)  # Shorter for test
    results = analyze_results(dbg)
    
    # Should have some hypotheses
    assert len(results["hypotheses"]) > 0
    # Should have couplings
    assert len(results["couplings"]) > 0
    # Should have events
    assert len(results["events"]) > 0
    
    # Check that hypotheses have confidence
    for hyp in results["hypotheses"]:
        assert 0.0 <= hyp.confidence <= 1.0


def test_demo_couplings_are_deduplicated():
    """Verify that coupled failures are deduplicated as per recent fixes."""
    model = create_failing_model()
    dataloader = create_problematic_data()
    
    dbg = train_with_monitoring(model, dataloader, num_steps=10)
    results = analyze_results(dbg)
    
    couplings = results["couplings"]
    
    # Check that couplings are unique pairs
    pair_set = set()
    for coupling in couplings:
        trigger = coupling.get("trigger", coupling.get("event1", "unknown"))
        consequence = coupling.get("consequence", coupling.get("event2", "unknown"))
        pair = (trigger, consequence)
        assert pair not in pair_set, f"Duplicate coupling found: {pair}"
        pair_set.add(pair)