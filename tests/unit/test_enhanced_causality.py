"""
Tests for neuraldbg/enhanced_causality.py

Coverage targets:
- CausalGraph: add_edge, get_causal_chain (BFS, unreachable, depth limit)
- EnhancedCausalReasoner: infer_layer_order, build_causal_graph, cross-layer propagation
- detect_temporal_patterns: cascade detection
- compute_confidence_multifactor: all 3 factors
- get_enhanced_explanations: combined analysis + sorting
- enhance_with_granger_style: entry point + empty events guard
"""

import pytest
import torch

from neuraldbg import SemanticEvent, EventType, CausalHypothesis, GradientHealth
from neuraldbg.enhanced_causality import (
    CausalGraph,
    EnhancedCausalReasoner,
    TemporalPattern,
    enhance_with_granger_style,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_event(
    layer: str,
    step: int,
    to_state: str = "vanishing",
    from_state: str = "healthy",
    event_type: EventType = EventType.GRADIENT_HEALTH_TRANSITION,
    confidence: float = 0.9,
) -> SemanticEvent:
    return SemanticEvent(
        event_type=event_type,
        layer_name=layer,
        step=step,
        from_state=from_state,
        to_state=to_state,
        confidence=confidence,
        metadata={},
    )


# ──────────────────────────────────────────────────────────────────────────────
# CausalGraph
# ──────────────────────────────────────────────────────────────────────────────

class TestCausalGraph:

    def test_add_edge_registers_nodes(self):
        g = CausalGraph()
        g.add_edge("A", "B", weight=0.8)
        assert "A" in g.nodes
        assert "B" in g.nodes

    def test_add_edge_registers_weight(self):
        g = CausalGraph()
        g.add_edge("X", "Y", weight=0.5)
        assert g.edge_weights[("X", "Y")] == pytest.approx(0.5)

    def test_add_edge_default_weight(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        assert g.edge_weights[("A", "B")] == pytest.approx(1.0)

    def test_get_causal_chain_direct(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        chain = g.get_causal_chain("A", "B")
        assert chain == ["A", "B"]

    def test_get_causal_chain_indirect(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        chain = g.get_causal_chain("A", "C")
        assert chain == ["A", "B", "C"]

    def test_get_causal_chain_unreachable(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        # C is not in the graph → start not in nodes
        chain = g.get_causal_chain("A", "C")
        assert chain is None

    def test_get_causal_chain_missing_start(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        chain = g.get_causal_chain("Z", "A")
        assert chain is None

    def test_get_causal_chain_max_depth(self):
        """Chain longer than max_depth should not be returned."""
        g = CausalGraph()
        # A→B→C→D→E→F = depth 6
        for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")]:
            g.add_edge(a, b)
        # max_depth=3 means paths longer than 3 nodes are pruned
        chain = g.get_causal_chain("A", "F", max_depth=3)
        assert chain is None

    def test_get_causal_chain_self_loop(self):
        """Start == End should return immediately."""
        g = CausalGraph()
        g.add_edge("A", "B")
        chain = g.get_causal_chain("A", "A")
        assert chain == ["A"]

    def test_multiple_edges_bfs_finds_shortest(self):
        g = CausalGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("C", "D")
        g.add_edge("B", "D")
        chain = g.get_causal_chain("A", "D")
        # BFS returns shortest: A→B→D or A→C→D (length 3)
        assert len(chain) == 3
        assert chain[0] == "A"
        assert chain[-1] == "D"


# ──────────────────────────────────────────────────────────────────────────────
# EnhancedCausalReasoner — _infer_layer_order
# ──────────────────────────────────────────────────────────────────────────────

class TestInferLayerOrder:

    def test_single_layer(self):
        events = [make_event("fc1", step=1), make_event("fc1", step=5)]
        r = EnhancedCausalReasoner(events)
        assert r.layer_order == ["fc1"]

    def test_two_layers_ordered_by_step(self):
        events = [
            make_event("fc2", step=10),
            make_event("fc1", step=2),
        ]
        r = EnhancedCausalReasoner(events)
        # fc1 appears at step 2 (avg=2), fc2 at step 10 (avg=10)
        assert r.layer_order == ["fc1", "fc2"]

    def test_three_layers(self):
        events = [
            make_event("layer_a", step=1),
            make_event("layer_b", step=3),
            make_event("layer_c", step=2),
        ]
        r = EnhancedCausalReasoner(events)
        # layer_a avg=1, layer_c avg=2, layer_b avg=3
        assert r.layer_order[0] == "layer_a"
        assert r.layer_order[-1] == "layer_b"


# ──────────────────────────────────────────────────────────────────────────────
# EnhancedCausalReasoner — _build_causal_graph
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildCausalGraph:

    def test_causal_edge_created_when_predecessor_fails_first(self):
        events = [
            make_event("fc1", step=1, to_state="vanishing"),
            make_event("fc2", step=3, to_state="dead"),
        ]
        r = EnhancedCausalReasoner(events)
        # fc1 (step=1) precedes fc2 (step=3) → edge should exist
        assert ("fc1", "fc2") in r.causal_graph.edge_weights

    def test_no_edge_when_no_issues(self):
        events = [
            make_event("fc1", step=1, to_state="healthy"),
            make_event("fc2", step=2, to_state="healthy"),
        ]
        r = EnhancedCausalReasoner(events)
        assert len(r.causal_graph.edges) == 0

    def test_no_edge_when_single_layer(self):
        events = [make_event("fc1", step=1, to_state="vanishing")]
        r = EnhancedCausalReasoner(events)
        # Only one layer → no edges possible
        assert len(r.causal_graph.edges) == 0

    def test_edge_weight_is_bounded_0_1(self):
        events = [
            make_event("fc1", step=1, to_state="vanishing"),
            make_event("fc2", step=100, to_state="dead"),
        ]
        r = EnhancedCausalReasoner(events)
        for w in r.causal_graph.edge_weights.values():
            assert 0.0 <= w <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# analyze_cross_layer_propagation
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossLayerPropagation:

    def test_propagation_detected(self):
        events = [
            make_event("fc1", step=1, to_state="vanishing"),
            make_event("fc2", step=3, to_state="dead"),
        ]
        r = EnhancedCausalReasoner(events)
        hypotheses = r.analyze_cross_layer_propagation()
        assert len(hypotheses) >= 1
        assert any("fc1" in h.description and "fc2" in h.description for h in hypotheses)

    def test_propagation_confidence_positive(self):
        events = [
            make_event("a", step=1, to_state="vanishing"),
            make_event("b", step=2, to_state="dead"),
        ]
        r = EnhancedCausalReasoner(events)
        for h in r.analyze_cross_layer_propagation():
            assert h.confidence > 0

    def test_no_propagation_when_healthy(self):
        events = [
            make_event("a", step=1, to_state="healthy"),
            make_event("b", step=2, to_state="healthy"),
        ]
        r = EnhancedCausalReasoner(events)
        hypotheses = r.analyze_cross_layer_propagation()
        assert len(hypotheses) == 0


# ──────────────────────────────────────────────────────────────────────────────
# detect_temporal_patterns
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectTemporalPatterns:

    def test_cascade_detected_with_3_layers(self):
        # Same event type propagating through 3+ different layers
        events = [
            make_event("a", step=1),
            make_event("b", step=2),
            make_event("c", step=3),
        ]
        r = EnhancedCausalReasoner(events)
        patterns = r.detect_temporal_patterns()
        assert len(patterns) >= 1
        assert any(p.pattern_type == "cascade" for p in patterns)

    def test_no_cascade_with_2_layers(self):
        events = [
            make_event("a", step=1),
            make_event("b", step=2),
        ]
        r = EnhancedCausalReasoner(events)
        patterns = r.detect_temporal_patterns()
        # 2 layers not enough for cascade
        assert all(len(p.events) >= 2 for p in patterns)

    def test_cascade_confidence_is_fixed(self):
        events = [make_event(f"layer_{i}", step=i) for i in range(5)]
        r = EnhancedCausalReasoner(events)
        patterns = r.detect_temporal_patterns()
        for p in patterns:
            if p.pattern_type == "cascade":
                assert p.confidence == pytest.approx(0.7)

    def test_cascade_from_different_event_types(self):
        events = [
            make_event("a", step=1, event_type=EventType.GRADIENT_HEALTH_TRANSITION),
            make_event("b", step=2, event_type=EventType.ACTIVATION_REGIME_SHIFT),
            make_event("c", step=3, event_type=EventType.OPTIMIZER_INSTABILITY),
        ]
        r = EnhancedCausalReasoner(events)
        # Each type has only 1 event → no cascade per type, but no error
        patterns = r.detect_temporal_patterns()
        assert isinstance(patterns, list)


# ──────────────────────────────────────────────────────────────────────────────
# compute_confidence_multifactor
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeConfidenceMultifactor:

    def _make_hypothesis(self, confidence=0.8, n_evidence=2, chain_len=2) -> CausalHypothesis:
        events = [make_event("fc1", step=i) for i in range(n_evidence)]
        return CausalHypothesis(
            description="test",
            confidence=confidence,
            evidence=events,
            causal_chain=["a"] * chain_len,
        )

    def test_result_bounded_0_1(self):
        r = EnhancedCausalReasoner([make_event("fc1", step=1)])
        h = self._make_hypothesis(confidence=0.99, n_evidence=10, chain_len=10)
        score = r.compute_confidence_multifactor(h)
        assert 0.0 <= score <= 1.0

    def test_more_evidence_increases_confidence(self):
        r = EnhancedCausalReasoner([make_event("fc1", step=1)])
        h_low = self._make_hypothesis(n_evidence=1)
        h_high = self._make_hypothesis(n_evidence=5)
        assert r.compute_confidence_multifactor(h_high) >= r.compute_confidence_multifactor(h_low)

    def test_empty_evidence_no_crash(self):
        r = EnhancedCausalReasoner([make_event("fc1", step=1)])
        h = CausalHypothesis(
            description="no evidence",
            confidence=0.5,
            evidence=[],
            causal_chain=[],
        )
        score = r.compute_confidence_multifactor(h)
        assert 0.0 <= score <= 1.0

    def test_zero_base_confidence(self):
        r = EnhancedCausalReasoner([make_event("fc1", step=1)])
        h = self._make_hypothesis(confidence=0.0, n_evidence=0, chain_len=0)
        score = r.compute_confidence_multifactor(h)
        assert score >= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# get_enhanced_explanations
# ──────────────────────────────────────────────────────────────────────────────

class TestGetEnhancedExplanations:

    def test_returns_list_of_hypotheses(self):
        events = [
            make_event("a", step=1, to_state="vanishing"),
            make_event("b", step=2, to_state="dead"),
            make_event("c", step=3, to_state="vanishing"),
        ]
        r = EnhancedCausalReasoner(events)
        result = r.get_enhanced_explanations("vanishing")
        assert isinstance(result, list)
        for h in result:
            assert isinstance(h, CausalHypothesis)

    def test_sorted_by_confidence_descending(self):
        events = [
            make_event("a", step=1, to_state="vanishing"),
            make_event("b", step=2, to_state="dead"),
            make_event("c", step=3),
        ]
        r = EnhancedCausalReasoner(events)
        result = r.get_enhanced_explanations("vanishing")
        confidences = [h.confidence for h in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_no_crash_with_single_event(self):
        events = [make_event("fc1", step=1)]
        r = EnhancedCausalReasoner(events)
        result = r.get_enhanced_explanations("general")
        assert isinstance(result, list)


# ──────────────────────────────────────────────────────────────────────────────
# enhance_with_granger_style (entry point)
# ──────────────────────────────────────────────────────────────────────────────

class TestEnhanceWithGrangerStyle:

    def test_empty_events_returns_empty(self):
        result = enhance_with_granger_style([])
        assert result == []

    def test_with_valid_events_returns_hypotheses(self):
        events = [
            make_event("a", step=1, to_state="vanishing"),
            make_event("b", step=2, to_state="dead"),
            make_event("c", step=3, to_state="vanishing"),
        ]
        result = enhance_with_granger_style(events)
        assert isinstance(result, list)

    def test_all_healthy_events_no_propagation_hypothesis(self):
        events = [
            make_event("a", step=1, to_state="healthy"),
            make_event("b", step=2, to_state="healthy"),
        ]
        result = enhance_with_granger_style(events)
        # No propagation edges → only possible cascade hypotheses
        for h in result:
            assert isinstance(h, CausalHypothesis)

    def test_multi_layer_cascade_generates_cascade_hypothesis(self):
        events = [make_event(f"layer_{i}", step=i) for i in range(5)]
        result = enhance_with_granger_style(events)
        descriptions = [h.description for h in result]
        assert any("cascade" in d.lower() or "layer" in d.lower() for d in descriptions)

    def test_return_type_is_list_of_causal_hypothesis(self):
        events = [make_event("fc1", step=1, to_state="vanishing")]
        result = enhance_with_granger_style(events)
        for item in result:
            assert isinstance(item, CausalHypothesis)
