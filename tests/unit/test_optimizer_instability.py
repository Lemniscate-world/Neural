"""
Unit tests for optimizer instability detection in NeuralDBG.

Tests for record_loss, _classify_optimizer_health, and _explain_optimizer_instability.
"""

import torch.nn as nn
from neuraldbg import (
    SemanticEvent,
    EventType,
    GradientHealth,
    OptimizerHealth,
    NeuralDbg,
)


class TestOptimizerHealthClassification:
    """Unit tests for _classify_optimizer_health method."""

    def test_stable_with_few_points(self):
        """Fewer than 3 loss values should always return STABLE."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.loss_history = [1.0, 0.9]
        assert dbg._classify_optimizer_health() == OptimizerHealth.STABLE

    def test_stable_normal_training(self):
        """Decreasing losses should be classified as STABLE."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.loss_history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        assert dbg._classify_optimizer_health() == OptimizerHealth.STABLE

    def test_diverging_nan(self):
        """NaN loss should be classified as DIVERGING."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.loss_history = [1.0, 0.9, 0.8, float("nan")]
        assert dbg._classify_optimizer_health() == OptimizerHealth.DIVERGING

    def test_diverging_inf(self):
        """Inf loss should be classified as DIVERGING."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.loss_history = [1.0, 0.9, 0.8, float("inf")]
        assert dbg._classify_optimizer_health() == OptimizerHealth.DIVERGING

    def test_loss_spike(self):
        """A sudden 10x increase in loss should be classified as LOSS_SPIKE."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.loss_history = [1.0, 0.9, 0.8, 0.7, 0.6, 100.0]
        assert dbg._classify_optimizer_health() == OptimizerHealth.LOSS_SPIKE

    def test_loss_plateau(self):
        """Constant loss values should be classified as LOSS_PLATEAU."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.loss_history = [0.5, 0.5, 0.5, 0.5, 0.5]
        assert dbg._classify_optimizer_health() == OptimizerHealth.LOSS_PLATEAU


class TestRecordLoss:
    """Unit tests for the record_loss method."""

    def test_record_loss_appends_to_history(self):
        """record_loss should add the value to loss_history."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.record_loss(1.0)
        dbg.record_loss(0.9)
        assert dbg.loss_history == [1.0, 0.9]

    def test_record_loss_emits_event_on_transition(self):
        """Transitioning to a non-stable state should emit an event."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        # Build up stable history
        for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
            dbg.record_loss(v)

        events_before = len(dbg.events)

        # Spike should trigger an event
        dbg.record_loss(100.0)

        instability_events = [
            e
            for e in dbg.events[events_before:]
            if e.event_type == EventType.OPTIMIZER_INSTABILITY
        ]
        assert len(instability_events) >= 1
        assert instability_events[0].to_state in (
            OptimizerHealth.LOSS_SPIKE.value,
            OptimizerHealth.DIVERGING.value,
        )

    def test_record_loss_no_event_when_stable(self):
        """No event should be emitted when loss stays stable."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        for v in [1.0, 0.95, 0.9, 0.85, 0.8]:
            dbg.record_loss(v)

        instability_events = [
            e for e in dbg.events if e.event_type == EventType.OPTIMIZER_INSTABILITY
        ]
        assert len(instability_events) == 0


class TestExplainOptimizerInstability:
    """Unit tests for _explain_optimizer_instability method."""

    def test_no_events_returns_empty(self):
        """No instability events should produce no hypotheses."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        assert len(dbg._explain_optimizer_instability()) == 0

    def test_spike_hypothesis(self):
        """A loss spike event should produce a hypothesis mentioning spike."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.events.append(
            SemanticEvent(
                event_type=EventType.OPTIMIZER_INSTABILITY,
                layer_name="optimizer",
                step=50,
                from_state=OptimizerHealth.STABLE.value,
                to_state=OptimizerHealth.LOSS_SPIKE.value,
                confidence=0.85,
                metadata={"recent_losses": [0.5, 0.5, 0.5, 0.5, 100.0]},
            )
        )

        hypotheses = dbg._explain_optimizer_instability()
        assert len(hypotheses) >= 1
        assert "spike" in hypotheses[0].description.lower()

    def test_diverging_with_gradient_explosion_cross_reference(self):
        """Diverging loss preceded by gradient explosion should produce a
        cross-referenced hypothesis with higher confidence."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Gradient explosion at step 45
        dbg.events.append(
            SemanticEvent(
                event_type=EventType.GRADIENT_HEALTH_TRANSITION,
                layer_name="linear1",
                step=45,
                from_state=GradientHealth.HEALTHY.value,
                to_state=GradientHealth.EXPLODING.value,
                confidence=0.9,
                metadata={},
            )
        )

        # Diverging loss at step 50
        dbg.events.append(
            SemanticEvent(
                event_type=EventType.OPTIMIZER_INSTABILITY,
                layer_name="optimizer",
                step=50,
                from_state=OptimizerHealth.STABLE.value,
                to_state=OptimizerHealth.DIVERGING.value,
                confidence=0.85,
                metadata={},
            )
        )

        hypotheses = dbg._explain_optimizer_instability()
        assert len(hypotheses) >= 2
        # The cross-referenced hypothesis should mention gradient explosion
        cross_ref = [h for h in hypotheses if "gradient" in h.description.lower()]
        assert len(cross_ref) >= 1
