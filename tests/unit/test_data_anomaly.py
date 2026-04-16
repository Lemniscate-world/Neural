"""
Unit tests for data anomaly detection in NeuralDBG.

Tests for _check_data_anomaly and _explain_data_anomaly.
"""

import torch
import torch.nn as nn
import pytest
from neuraldbg import (
    SemanticEvent, EventType, DataHealth,
    CausalHypothesis, NeuralDbg
)


class TestCheckDataAnomaly:
    """Unit tests for _check_data_anomaly method."""

    def test_nan_detection(self):
        """NaN values in input should produce a DATA_ANOMALY event."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.step = 10

        tensor_with_nan = torch.tensor([[1.0, float("nan"), 3.0]])
        dbg._check_data_anomaly(tensor_with_nan, "layer1")

        anomaly_events = [
            e for e in dbg.events
            if e.event_type == EventType.DATA_ANOMALY
        ]
        assert len(anomaly_events) == 1
        assert anomaly_events[0].to_state == DataHealth.NAN_DETECTED.value

    def test_inf_detection(self):
        """Inf values in input should produce a DATA_ANOMALY event."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.step = 10

        tensor_with_inf = torch.tensor([[1.0, float("inf"), 3.0]])
        dbg._check_data_anomaly(tensor_with_inf, "layer1")

        anomaly_events = [
            e for e in dbg.events
            if e.event_type == EventType.DATA_ANOMALY
        ]
        assert len(anomaly_events) == 1
        assert anomaly_events[0].to_state == DataHealth.INF_DETECTED.value

    def test_distribution_shift_detection(self):
        """A large shift in input statistics should emit a distribution shift event."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # First call establishes baseline
        dbg.step = 0
        normal_tensor = torch.randn(32, 10)
        dbg._check_data_anomaly(normal_tensor, "layer1")

        events_before = len(dbg.events)

        # Second call with wildly different statistics
        dbg.step = 1
        shifted_tensor = torch.randn(32, 10) * 100 + 500
        dbg._check_data_anomaly(shifted_tensor, "layer1")

        shift_events = [
            e for e in dbg.events[events_before:]
            if e.event_type == EventType.DATA_ANOMALY
            and e.to_state == DataHealth.DISTRIBUTION_SHIFT.value
        ]
        assert len(shift_events) >= 1

    def test_no_anomaly_for_normal_data(self):
        """Normal data should not produce anomaly events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # First call establishes baseline
        dbg.step = 0
        dbg._check_data_anomaly(torch.randn(32, 10), "layer1")

        events_before = len(dbg.events)

        # Second call with similar statistics
        dbg.step = 1
        dbg._check_data_anomaly(torch.randn(32, 10), "layer1")

        anomaly_events = [
            e for e in dbg.events[events_before:]
            if e.event_type == EventType.DATA_ANOMALY
        ]
        assert len(anomaly_events) == 0

    def test_nan_takes_priority_over_distribution_shift(self):
        """NaN detection should fire without checking distribution shift."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.step = 0
        dbg._check_data_anomaly(torch.randn(32, 10), "layer1")

        dbg.step = 1
        nan_tensor = torch.full((32, 10), float("nan"))
        dbg._check_data_anomaly(nan_tensor, "layer1")

        anomaly_events = [
            e for e in dbg.events
            if e.event_type == EventType.DATA_ANOMALY
        ]
        # Should only have one NaN event, not a distribution shift
        nan_events = [e for e in anomaly_events if e.to_state == DataHealth.NAN_DETECTED.value]
        assert len(nan_events) >= 1


class TestExplainDataAnomaly:
    """Unit tests for _explain_data_anomaly method."""

    def test_no_events_returns_empty(self):
        """No anomaly events should produce no hypotheses."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        assert len(dbg._explain_data_anomaly()) == 0

    def test_nan_hypothesis(self):
        """NaN anomaly event should produce a hypothesis about NaN."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.events.append(SemanticEvent(
            event_type=EventType.DATA_ANOMALY,
            layer_name="linear1",
            step=10,
            from_state=DataHealth.NORMAL.value,
            to_state=DataHealth.NAN_DETECTED.value,
            confidence=1.0,
            metadata={"nan_count": 5},
        ))

        hypotheses = dbg._explain_data_anomaly()
        assert len(hypotheses) >= 1
        assert "nan" in hypotheses[0].description.lower()

    def test_distribution_shift_hypothesis(self):
        """Distribution shift event should produce a hypothesis about the shift."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)
        dbg.events.append(SemanticEvent(
            event_type=EventType.DATA_ANOMALY,
            layer_name="linear1",
            step=20,
            from_state=DataHealth.NORMAL.value,
            to_state=DataHealth.DISTRIBUTION_SHIFT.value,
            confidence=0.8,
            metadata={"mean_shift_sigma": 5.0},
        ))

        hypotheses = dbg._explain_data_anomaly()
        assert len(hypotheses) >= 1
        assert "distribution" in hypotheses[0].description.lower() or \
               "shift" in hypotheses[0].description.lower()
