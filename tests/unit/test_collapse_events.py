"""
Unit tests for event collapsing in NeuralDBG.

Tests for the _collapse_events method which merges sequential events
in the same layer into summary traces.
"""

import torch.nn as nn
import pytest
from neuraldbg import (
    SemanticEvent, EventType, GradientHealth, ActivationHealth,
    NeuralDbg
)


class TestCollapseEvents:
    """Unit tests for _collapse_events method."""

    def test_empty_events(self):
        """Empty event list should return empty."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        assert dbg._collapse_events() == []

    def test_single_event_unchanged(self):
        """A single event should be returned as-is."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        event = SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={},
        )
        dbg.events.append(event)
        collapsed = dbg._collapse_events()
        assert len(collapsed) == 1
        assert collapsed[0].from_state == GradientHealth.HEALTHY.value

    def test_chain_is_collapsed(self):
        """A -> B -> C in the same layer should be collapsed to A -> C."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.SATURATED.value,
            confidence=0.7,
            metadata={"prev_norm": 1.0},
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=20,
            from_state=GradientHealth.SATURATED.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={"prev_norm": 0.001},
        ))

        collapsed = dbg._collapse_events()
        assert len(collapsed) == 1
        assert collapsed[0].from_state == GradientHealth.HEALTHY.value
        assert collapsed[0].to_state == GradientHealth.VANISHING.value
        assert collapsed[0].metadata.get("collapsed_count") == 2
        assert "10-20" in collapsed[0].metadata.get("step_range", "")

    def test_reverted_chain_not_collapsed(self):
        """A -> B -> A should NOT be collapsed (states reverted)."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.SATURATED.value,
            confidence=0.7,
            metadata={},
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=20,
            from_state=GradientHealth.SATURATED.value,
            to_state=GradientHealth.HEALTHY.value,
            confidence=0.8,
            metadata={},
        ))

        collapsed = dbg._collapse_events()
        # States reverted: A -> B -> A, so from_state == to_state, keep all
        assert len(collapsed) == 2

    def test_different_layers_not_collapsed(self):
        """Events in different layers should not be collapsed together."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={},
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear2",
            step=15,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.EXPLODING.value,
            confidence=0.8,
            metadata={},
        ))

        collapsed = dbg._collapse_events()
        assert len(collapsed) == 2

    def test_different_event_types_not_collapsed(self):
        """Events of different types in the same layer should not collapse."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={},
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name="linear1",
            step=12,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.DEAD.value,
            confidence=0.85,
            metadata={},
        ))

        collapsed = dbg._collapse_events()
        assert len(collapsed) == 2

    def test_baseline_events_never_collapsed(self):
        """Baseline events (from_state='NONE') should always be kept individually."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Baseline event (initial state capture)
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=0,
            from_state="NONE",
            to_state=GradientHealth.HEALTHY.value,
            confidence=1.0,
            metadata={},
        ))
        # Transition event in same layer
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={},
        ))

        collapsed = dbg._collapse_events()
        # Baseline kept individually + transition kept individually = 2 events
        assert len(collapsed) == 2
        baseline = [e for e in collapsed if e.from_state == "NONE"]
        assert len(baseline) == 1

    def test_baseline_does_not_break_revert_detection(self):
        """Baseline events should not interfere with revert detection on transitions."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Baseline
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=0,
            from_state="NONE",
            to_state=GradientHealth.HEALTHY.value,
            confidence=1.0,
            metadata={},
        ))
        # A -> B
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.SATURATED.value,
            confidence=0.7,
            metadata={},
        ))
        # B -> A (reversion)
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=20,
            from_state=GradientHealth.SATURATED.value,
            to_state=GradientHealth.HEALTHY.value,
            confidence=0.8,
            metadata={},
        ))

        collapsed = dbg._collapse_events()
        # Baseline (1) + 2 reverted transitions kept individually = 3
        assert len(collapsed) == 3

    def test_collapsed_uses_max_confidence(self):
        """Collapsed event should use the maximum confidence from the chain."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=10,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.SATURATED.value,
            confidence=0.5,
            metadata={},
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=20,
            from_state=GradientHealth.SATURATED.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.95,
            metadata={},
        ))

        collapsed = dbg._collapse_events()
        assert len(collapsed) == 1
        assert collapsed[0].confidence == 0.95
