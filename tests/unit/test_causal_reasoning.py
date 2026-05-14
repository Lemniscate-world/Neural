"""
Unit tests for causal reasoning methods in NeuralDBG.

Tests for _explain_exploding_gradients, _explain_dead_neurons,
_explain_saturated_activations, and export_mermaid_causal_graph.
"""

import re

import torch
import torch.nn as nn
import pytest
from neuraldbg import (
    SemanticEvent, EventType, GradientHealth, ActivationHealth, CausalHypothesis,
    NeuralDbg
)


class TestExplodingGradients:
    """Unit tests for _explain_exploding_gradients method."""

    def test_explain_exploding_no_events(self):
        """Test explaining exploding gradients when no events exist."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        hypotheses = dbg._explain_exploding_gradients()
        assert len(hypotheses) == 0

    def test_explain_exploding_with_events(self):
        """Test explaining exploding gradients with detected events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add an exploding gradient event
        event = SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=100,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.EXPLODING.value,
            confidence=0.95,
            metadata={'prev_norm': 1.0, 'current_norm': 1e5}
        )
        dbg.events.append(event)

        hypotheses = dbg._explain_exploding_gradients()
        assert len(hypotheses) >= 1
        assert "explosion originated" in hypotheses[0].description.lower()
        assert hypotheses[0].confidence == 0.95

    def test_explain_exploding_multiple_events(self):
        """Test that only first exploding event is used for hypothesis."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add multiple exploding events
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.EXPLODING.value,
            confidence=0.9,
            metadata={}
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear2',
            step=100,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.EXPLODING.value,
            confidence=0.8,
            metadata={}
        ))

        hypotheses = dbg._explain_exploding_gradients()
        assert len(hypotheses) >= 1
        # Should use the first event (step 50)
        assert "step 50" in hypotheses[0].description


class TestDeadNeurons:
    """Unit tests for _explain_dead_neurons method."""

    def test_explain_dead_no_events(self):
        """Test explaining dead neurons when no events exist."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        hypotheses = dbg._explain_dead_neurons()
        assert len(hypotheses) == 0

    def test_explain_dead_with_events(self):
        """Test explaining dead neurons with detected events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add a dead neuron event (high dead_ratio)
        event = SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='relu1',
            step=200,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.DEAD.value,
            confidence=0.88,
            metadata={}
        )
        dbg.events.append(event)

        hypotheses = dbg._explain_dead_neurons()
        assert len(hypotheses) >= 1
        assert "neuron death" in hypotheses[0].description.lower()
        assert hypotheses[0].confidence == 0.88

    def test_explain_dead_low_ratio_ignored(self):
        """Test that events with low dead_ratio are ignored."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add event with low dead_ratio (should be ignored)
        dbg.events.append(SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='relu1',
            step=200,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.NORMAL.value,  # Low ratio doesn't trigger shift in new logic
            confidence=0.7,
            metadata={}
        ))

        hypotheses = dbg._explain_dead_neurons()
        assert len(hypotheses) == 0


class TestSaturatedActivations:
    """Unit tests for _explain_saturated_activations method."""

    def test_explain_saturated_no_events(self):
        """Test explaining saturated activations when no events exist."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        hypotheses = dbg._explain_saturated_activations()
        assert len(hypotheses) == 0

    def test_explain_saturated_with_events(self):
        """Test explaining saturated activations with detected events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add a saturation event
        event = SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='sigmoid1',
            step=150,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.SATURATED.value,
            confidence=0.92,
            metadata={}
        )
        dbg.events.append(event)

        hypotheses = dbg._explain_saturated_activations()
        assert len(hypotheses) >= 1
        assert "saturation" in hypotheses[0].description.lower()
        assert hypotheses[0].confidence == 0.92

    def test_explain_saturated_uses_baseline_saturation_ratio(self):
        """Test saturation explanation reports baseline metadata accurately."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        event = SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='sigmoid1',
            step=0,
            from_state="NONE",
            to_state=ActivationHealth.SATURATED.value,
            confidence=1.0,
            metadata={"saturation_ratio": 0.84}
        )
        dbg.events.append(event)

        hypotheses = dbg._explain_saturated_activations()

        assert "0.84" in hypotheses[0].causal_chain[0]

    def test_explain_saturated_low_ratio_ignored(self):
        """Test that events with low saturation_ratio are ignored."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add event with low saturation_ratio (should be ignored)
        dbg.events.append(SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='sigmoid1',
            step=150,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.NORMAL.value,
            confidence=0.6,
            metadata={}
        ))

        hypotheses = dbg._explain_saturated_activations()
        assert len(hypotheses) == 0


class TestExplainFailure:
    """Unit tests for explain_failure method."""

    def test_explain_failure_vanishing(self):
        """Test explain_failure with vanishing_gradients type."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={}
        ))

        hypotheses = dbg.explain_failure("vanishing_gradients")
        assert len(hypotheses) >= 1
        assert hypotheses[0].confidence == 0.9

    def test_vanishing_coupling_uses_only_saturated_activation(self):
        """Vanishing explanations should not cite normal activations as saturation."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.extend([
            SemanticEvent(
                event_type=EventType.ACTIVATION_REGIME_SHIFT,
                layer_name="relu1",
                step=0,
                from_state="NONE",
                to_state=ActivationHealth.NORMAL.value,
                confidence=1.0,
                metadata={},
            ),
            SemanticEvent(
                event_type=EventType.GRADIENT_HEALTH_TRANSITION,
                layer_name="linear1",
                step=0,
                from_state="NONE",
                to_state=GradientHealth.VANISHING.value,
                confidence=1.0,
                metadata={},
            ),
        ])

        hypotheses = dbg._explain_vanishing_gradients()

        assert len(hypotheses) == 1
        assert "activation mismatch" not in hypotheses[0].description

    def test_explain_failure_exploding(self):
        """Test explain_failure with exploding_gradients type."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.EXPLODING.value,
            confidence=0.85,
            metadata={}
        ))

        hypotheses = dbg.explain_failure("exploding_gradients")
        assert len(hypotheses) >= 1

    def test_explain_failure_dead_neurons(self):
        """Test explain_failure with dead_neurons type."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='relu1',
            step=100,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.DEAD.value,
            confidence=0.8,
            metadata={}
        ))

        hypotheses = dbg.explain_failure("dead_neurons")
        assert len(hypotheses) >= 1

    def test_explain_failure_saturated(self):
        """Test explain_failure with saturated_activations type."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='sigmoid1',
            step=100,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.SATURATED.value,
            confidence=0.75,
            metadata={}
        ))

        hypotheses = dbg.explain_failure("saturated_activations")
        assert len(hypotheses) >= 1

    def test_explain_failure_sorted_by_confidence(self):
        """Test that hypotheses are sorted by confidence."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add multiple events with different confidences
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.5,
            metadata={}
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear2',
            step=100,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={}
        ))

        hypotheses = dbg.explain_failure("vanishing_gradients")
        # Should be sorted by confidence (highest first)
        if len(hypotheses) > 1:
            assert hypotheses[0].confidence >= hypotheses[1].confidence


class TestMermaidGraphExport:
    """Unit tests for export_mermaid_causal_graph method."""

    def test_export_empty_events(self):
        """Test export with no events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        graph = dbg.export_mermaid_causal_graph()
        assert "graph TD" in graph

    def test_export_with_events(self):
        """Test export with events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={}
        ))

        graph = dbg.export_mermaid_causal_graph()
        assert "graph TD" in graph
        assert "E0" in graph
        assert "gradient_health_transition" in graph
        assert "linear1" in graph

    def test_export_with_coupled_events(self):
        """Test export with coupled events shows coupling edge."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add coupled events (close in time, different layers)
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={}
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name='relu1',
            step=52,
            from_state=ActivationHealth.NORMAL.value,
            to_state=ActivationHealth.DEAD.value,
            confidence=0.8,
            metadata={}
        ))

        graph = dbg.export_mermaid_causal_graph()
        assert "coupled" in graph

    def test_export_deduplicates_coupled_edges(self):
        """Duplicate logical couplings should render one Mermaid coupled edge."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.extend([
            SemanticEvent(
                event_type=EventType.ACTIVATION_REGIME_SHIFT,
                layer_name='tanh1',
                step=10,
                from_state=ActivationHealth.NORMAL.value,
                to_state=ActivationHealth.SATURATED.value,
                confidence=0.6,
                metadata={},
            ),
            SemanticEvent(
                event_type=EventType.ACTIVATION_REGIME_SHIFT,
                layer_name='tanh1',
                step=11,
                from_state=ActivationHealth.NORMAL.value,
                to_state=ActivationHealth.SATURATED.value,
                confidence=0.9,
                metadata={},
            ),
            SemanticEvent(
                event_type=EventType.GRADIENT_HEALTH_TRANSITION,
                layer_name='linear2',
                step=12,
                from_state=GradientHealth.HEALTHY.value,
                to_state=GradientHealth.VANISHING.value,
                confidence=0.8,
                metadata={},
            ),
        ])

        graph = dbg.export_mermaid_causal_graph()

        # Two activation SemanticEvents on tanh1 produce one logical
        # tanh1-to-linear2 coupling in the exported graph.
        assert len(re.findall(r"-+>\s*\|coupled\|", graph)) == 1

    def test_export_temporal_flow(self):
        """Test export shows temporal flow for same layer."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        # Add events in same layer at different steps
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={}
        ))
        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=100,
            from_state=GradientHealth.VANISHING.value,
            to_state=GradientHealth.HEALTHY.value,
            confidence=0.8,
            metadata={}
        ))

        graph = dbg.export_mermaid_causal_graph()
        assert "temporal" in graph


class TestTraceCausalChain:
    """Unit tests for trace_causal_chain method."""

    def test_trace_empty(self):
        """Test trace with no events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        chain = dbg.trace_causal_chain("gradient_health_transition")
        assert len(chain) == 0

    def test_trace_with_events(self):
        """Test trace with matching events."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={'test': 'value'}
        ))

        chain = dbg.trace_causal_chain("gradient_health_transition")
        assert len(chain) == 1
        assert "linear1" in chain[0]
        assert "step 50" in chain[0]


class TestGetCausalHypotheses:
    """Unit tests for get_causal_hypotheses method."""

    def test_get_causal_hypotheses_default(self):
        """Test get_causal_hypotheses returns vanishing by default."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        dbg.events.append(SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name='linear1',
            step=50,
            from_state=GradientHealth.HEALTHY.value,
            to_state=GradientHealth.VANISHING.value,
            confidence=0.9,
            metadata={}
        ))

        hypotheses = dbg.get_causal_hypotheses()
        assert len(hypotheses) >= 1


class TestRootCauseEvidence:
    """Unit tests for first-occurrence root cause evidence selection."""

    def test_root_cause_uses_event_matching_failure_type(self):
        """Root causes should not attach unrelated same-layer same-step evidence."""
        model = nn.Linear(10, 5)
        dbg = NeuralDbg(model)

        activation_event = SemanticEvent(
            event_type=EventType.ACTIVATION_REGIME_SHIFT,
            layer_name="linear1",
            step=0,
            from_state="NONE",
            to_state=ActivationHealth.NORMAL.value,
            confidence=1.0,
            metadata={},
        )
        gradient_event = SemanticEvent(
            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
            layer_name="linear1",
            step=0,
            from_state="NONE",
            to_state=GradientHealth.VANISHING.value,
            confidence=1.0,
            metadata={},
        )
        dbg.events.extend([activation_event, gradient_event])
        dbg.first_failure_layer["gradient_vanishing"] = "linear1"
        dbg.first_failure_step["gradient_vanishing"] = 0

        hypotheses = dbg.get_root_causes()

        assert len(hypotheses) == 1
        assert hypotheses[0].evidence == [gradient_event]
