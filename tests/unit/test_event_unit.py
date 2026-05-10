"""
Unit tests for the SemanticEvent and NeuralDbg classes in NeuralDBG.

These tests focus on the causal inference engine components,
ensuring semantic event extraction and causal reasoning work correctly.
"""

import torch
import torch.nn as nn
import pytest
from neuraldbg import (
    SemanticEvent, EventType, GradientHealth, ActivationHealth, CausalHypothesis,
    NeuralDbg
)


class TestSemanticEvent:
    """Unit tests for SemanticEvent class."""

    def test_semantic_event_creation(self):
        """Test that SemanticEvent is created with correct attributes."""
        event = SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=42,
            from_state=GradientHealth.HEALTHY,
            to_state=GradientHealth.VANISHING,
            confidence=0.85,
            metadata={'prev_norm': 1.0, 'current_norm': 1e-8}
        )

        assert event.event_type == EventType.GRADIENT_HEALTH_TRANSITION
        assert event.layer_name == 'linear1'
        assert event.step == 42
        assert event.from_state == GradientHealth.HEALTHY
        assert event.to_state == GradientHealth.VANISHING
        assert event.confidence == 0.85
        assert event.metadata['prev_norm'] == 1.0


class TestNeuralDbgCore:
    """Unit tests for core NeuralDbg functionality."""

    def test_initialization(self):
        """Test NeuralDbg initialization."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        assert dbg.model is model
        assert len(dbg.events) == 0
        assert not dbg.is_monitoring
        assert len(dbg.hooks) == 0

    def test_gradient_health_classification(self):
        """Test gradient health classification."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Test healthy gradient
        assert dbg._classify_gradient_health(1.0) == GradientHealth.HEALTHY

        # Test vanishing gradient
        assert dbg._classify_gradient_health(1e-8) == GradientHealth.VANISHING

        # Test exploding gradient
        assert dbg._classify_gradient_health(1e4) == GradientHealth.EXPLODING

        # Test saturated gradient (Small but > vanishing)
        assert dbg._classify_gradient_health(dbg.threshold_vanishing * 10) == GradientHealth.SATURATED

    def test_activation_stats_computation(self):
        """Test activation statistics computation."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Create test tensor
        tensor = torch.tensor([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]], dtype=torch.float32)
        stats = dbg._compute_activation_stats(tensor)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'sparsity' in stats  # Fraction of zeros
        assert 'norm' in stats

        assert stats['min'] == -1.0
        assert stats['max'] == 3.0
        assert stats['sparsity'] == pytest.approx(2/6)  # 2 zeros out of 6 elements

    def test_gradient_transition_detection(self):
        """Test detection of gradient health transitions."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Healthy to vanishing transition
        transition = dbg._detect_gradient_transition(1.0, 1e-8)
        assert transition is not None
        assert 'type' in transition
        assert transition['type'] == 'healthy_to_vanishing'
        assert transition['confidence'] > 0

        # No transition (both healthy)
        transition = dbg._detect_gradient_transition(1.0, 0.8)
        assert transition is None

    def test_activation_shift_detection(self):
        """Test detection of activation regime shifts via semantic states."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Normal to Dead transition
        prev_stats = {'dead_ratio': 0.1, 'saturation_ratio': 0.1, 'std': 1.0}
        current_stats = {'dead_ratio': 0.95, 'saturation_ratio': 0.1, 'std': 0.0} 

        shift = dbg._detect_activation_shift(prev_stats, current_stats)
        assert shift is not None
        assert 'dead' in shift['type']
        assert shift['confidence'] == 0.9


class TestCausalReasoning:
    """Unit tests for causal reasoning functionality."""

    def test_causal_hypothesis_creation(self):
        """Test CausalHypothesis creation."""
        event = SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=42,
            from_state=GradientHealth.HEALTHY,
            to_state=GradientHealth.VANISHING,
            confidence=0.85,
            metadata={}
        )

        hypothesis = CausalHypothesis(
            description="Test hypothesis",
            confidence=0.85,
            evidence=[event],
            causal_chain=["Step 1", "Step 2"]
        )

        assert hypothesis.description == "Test hypothesis"
        assert hypothesis.confidence == 0.85
        assert len(hypothesis.evidence) == 1
        assert hypothesis.causal_chain == ["Step 1", "Step 2"]

    def test_explain_vanishing_gradients_no_events(self):
        """Test explaining vanishing gradients when no events exist."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        hypotheses = dbg._explain_vanishing_gradients()
        assert len(hypotheses) == 0

    def test_explain_vanishing_gradients_with_events(self):
        """Test explaining vanishing gradients with detected events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Manually add a vanishing gradient event
        event = SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={'prev_norm': 1.0, 'current_norm': 1e-8}
        )
        dbg.events.append(event)

        hypotheses = dbg._explain_vanishing_gradients()
        assert len(hypotheses) >= 1
        assert "originated in layer 'linear1'" in hypotheses[0].description
        assert hypotheses[0].confidence == 0.9

    def test_detect_coupled_failures(self):
        """Test detection of coupled failures."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add events close in time but different layers
        event1 = SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY,
            to_state=GradientHealth.VANISHING,
            confidence=0.8,
            metadata={}
        )
        event2 = SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='relu1',
            step=52,  # Close in time
            from_state={'sparsity': 0.1},
            to_state={'sparsity': 0.5},
            confidence=0.7,
            metadata={}
        )
        dbg.events.extend([event1, event2])

        couplings = dbg.detect_coupled_failures()
        assert len(couplings) >= 1
        coupling = couplings[0]
        assert 'linear1' in coupling['trigger']
        assert 'relu1' in coupling['consequence']
        assert coupling['step_difference'] == 2


class TestResourceProfiling:
    """Unit tests for resource profiling integration in NeuralDbg."""

    def test_sample_resources_cpu(self, monkeypatch):
        """cpu_memory_mb is populated when psutil is available."""
        import types
        model = nn.Linear(4, 2)
        dbg = NeuralDbg(model)

        # Monkeypatch a fake psutil process on the instance
        fake_mem = types.SimpleNamespace(rss=200 * 1024 ** 2)  # 200 MB
        fake_proc = types.SimpleNamespace(memory_info=lambda: fake_mem)
        dbg._psutil_process = fake_proc

        stats = dbg._sample_resources()
        assert 'cpu_memory_mb' in stats
        assert stats['cpu_memory_mb'] == pytest.approx(200.0)

    def test_sample_resources_no_psutil(self):
        """Sampling degrades gracefully when psutil is unavailable."""
        model = nn.Linear(4, 2)
        dbg = NeuralDbg(model)
        dbg._psutil_process = None  # simulate absent psutil

        stats = dbg._sample_resources()
        assert 'cpu_memory_mb' not in stats

    def test_memory_spike_detected(self):
        """_is_memory_spike returns True when rise > 20 % and > 50 MB."""
        model = nn.Linear(4, 2)
        dbg = NeuralDbg(model)

        baseline = {'cpu_memory_mb': 200.0}
        current = {'cpu_memory_mb': 200.0 + 60.0}  # +60 MB = 30 % rise
        is_spike, keys = dbg._is_memory_spike(current, baseline)
        assert is_spike is True
        assert 'cpu_memory_mb' in keys

    def test_memory_spike_not_triggered_below_threshold(self):
        """No spike when rise is < 50 MB even if percentage is high."""
        model = nn.Linear(4, 2)
        dbg = NeuralDbg(model)

        baseline = {'cpu_memory_mb': 10.0}
        current = {'cpu_memory_mb': 25.0}  # +15 MB — > 20 % but < 50 MB
        is_spike, keys = dbg._is_memory_spike(current, baseline)
        assert is_spike is False
        assert keys == []

    def test_event_metadata_contains_resources(self, monkeypatch):
        """SemanticEvents emitted by hooks carry resources/memory_spike keys."""
        import types
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        dbg = NeuralDbg(model)

        fake_mem = types.SimpleNamespace(rss=300 * 1024 ** 2)
        dbg._psutil_process = types.SimpleNamespace(memory_info=lambda: fake_mem)

        x = torch.randn(2, 4)
        with dbg:
            for step in range(3):
                dbg.step = step
                out = model(x)
                loss = out.sum()
                loss.backward()

        for event in dbg.events:
            assert 'resources' in event.metadata
            assert 'memory_spike' in event.metadata
            assert 'memory_spike_keys' in event.metadata
            assert isinstance(event.metadata['memory_spike'], bool)

    def test_step_snapshot_cached(self, monkeypatch):
        """Resource snapshot is taken only once per step, not once per module."""
        import types
        call_count = {'n': 0}
        model = nn.Linear(4, 2)
        dbg = NeuralDbg(model)

        original_sample = dbg._sample_resources

        def counting_sample(device=None):
            call_count['n'] += 1
            return original_sample(device)

        monkeypatch.setattr(dbg, '_sample_resources', counting_sample)

        dbg.step = 0
        dbg._get_step_resource_snapshot()
        dbg._get_step_resource_snapshot()  # same step — should not call again
        assert call_count['n'] == 1

        dbg.step = 1
        dbg._get_step_resource_snapshot()  # new step — should call once more
        assert call_count['n'] == 2


class TestIntegrationBasics:
    """Basic integration tests for NeuralDbg with simple models."""

    def test_context_manager(self):
        """Test that NeuralDbg works as a context manager."""
        model = nn.Linear(10, 5)
        with NeuralDbg(model) as dbg:
            assert dbg.is_monitoring
            assert len(dbg.hooks) > 0  # Hooks should be installed

        # After exiting context, should be cleaned up
        assert not dbg.is_monitoring
        assert len(dbg.hooks) == 0

    def test_hook_installation(self):
        """Test that hooks are properly installed on model modules."""
        model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 5)
        )
        dbg = NeuralDbg(model)

        with dbg:
            # Should have hooks on each module (forward + backward)
            expected_hooks = 4 * 2  # 3 sub-modules + 1 root × 2 hooks each
            assert len(dbg.hooks) == expected_hooks
