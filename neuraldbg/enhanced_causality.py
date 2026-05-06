"""
Enhanced Causal Reasoning Module
===================================
Phase 2 Enhancement: Advanced causal inference beyond pattern matching.

Features:
- Granger-style causality (temporal precedence)
- Cross-layer propagation tracking
- Multi-factor confidence scoring
- Bayesian-inspired belief updates
"""

from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from neuraldbg import SemanticEvent, EventType, CausalHypothesis


@dataclass
class CausalGraph:
    """Directed graph representing causal relationships between events."""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    edge_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    def add_edge(self, from_node: str, to_node: str, weight: float = 1.0):
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges[from_node].append(to_node)
        self.edge_weights[(from_node, to_node)] = weight
    
    def get_causal_chain(self, start: str, end: str, max_depth: int = 5) -> Optional[List[str]]:
        """Find causal chain from start to end using BFS."""
        if start not in self.nodes or end not in self.nodes:
            return None
        
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.edges.get(current, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None


@dataclass  
class TemporalPattern:
    """Detected temporal pattern in event sequence."""
    pattern_type: str  # "cascade", "coupled", "cyclic"
    confidence: float
    events: List[SemanticEvent]
    description: str


class EnhancedCausalReasoner:
    """
    Enhanced causal reasoning with:
    - Temporal precedence analysis (Granger-style)
    - Cross-layer propagation tracking
    - Multi-factor confidence scoring
    """
    
    def __init__(self, events: List[SemanticEvent]):
        self.events = events
        self.causal_graph = CausalGraph()
        self.layer_order = self._infer_layer_order()
        self._build_causal_graph()
    
    def _infer_layer_order(self) -> List[str]:
        """Infer layer ordering from event sequence patterns."""
        layer_steps = defaultdict(list)
        for e in self.events:
            layer_steps[e.layer_name].append(e.step)
        
        # Sort by average first appearance
        avg_steps = {layer: np.mean(steps) for layer, steps in layer_steps.items()}
        return sorted(avg_steps.keys(), key=lambda x: avg_steps[x])
    
    def _build_causal_graph(self):
        """Build directed graph of causal relationships."""
        events_by_layer = defaultdict(list)
        for e in self.events:
            events_by_layer[e.layer_name].append(e)
        
        for layer_idx, layer in enumerate(self.layer_order[:-1]):
            next_layer = self.layer_order[layer_idx + 1]
            current_events = events_by_layer[layer]
            next_events = events_by_layer[next_layer]
            
            # Check if events in current layer precede issues in next layer
            current_issues = [e for e in current_events 
                            if e.to_state in ['vanishing', 'dead', 'saturated']]
            next_issues = [e for e in next_events
                          if e.to_state in ['vanishing', 'dead', 'saturated']]
            
            if current_issues and next_issues:
                # Check temporal precedence
                current_first = min(e.step for e in current_issues)
                next_first = min(e.step for e in next_issues)
                
                if current_first <= next_first:
                    # Causal relationship likely
                    weight = min(1.0, (next_first - current_first + 1) / 10)
                    self.causal_graph.add_edge(layer, next_layer, weight)
    
    def analyze_cross_layer_propagation(self) -> List[CausalHypothesis]:
        """Analyze how failures propagate through layers."""
        hypotheses = []
        
        for i, layer in enumerate(self.layer_order[:-1]):
            next_layer = self.layer_order[i + 1]
            chain = self.causal_graph.get_causal_chain(layer, next_layer)
            
            if chain and len(chain) >= 2:
                # Find events that represent this propagation
                layer_events = [e for e in self.events if e.layer_name == layer]
                next_events = [e for e in self.events if e.layer_name == next_layer]
                
                if layer_events and next_events:
                    # Confidence based on chain strength
                    edge_weight = self.causal_graph.edge_weights.get((layer, next_layer), 0.5)
                    
                    hypotheses.append(CausalHypothesis(
                        description=f"Failure propagation: '{layer}' -> '{next_layer}' through causal chain {chain}",
                        confidence=edge_weight * 0.9,
                        evidence=layer_events[:1] + next_events[:1],
                        causal_chain=chain
                    ))
        
        return hypotheses
    
    def detect_temporal_patterns(self) -> List[TemporalPattern]:
        """Detect temporal patterns in event sequence."""
        patterns = []
        
        # Group events by type
        events_by_type = defaultdict(list)
        for e in self.events:
            events_by_type[e.event_type.value].append(e)
        
        # Detect cascades (same type, sequential layers)
        for event_type, events in events_by_type.items():
            events_sorted = sorted(events, key=lambda e: e.step)
            
            # Check for cascade pattern
            layers_seen = []
            for e in events_sorted:
                if e.layer_name not in layers_seen:
                    layers_seen.append(e.layer_name)
                    
                    if len(layers_seen) >= 3:
                        patterns.append(TemporalPattern(
                            pattern_type="cascade",
                            confidence=0.7,
                            events=events_sorted[:5],
                            description=f"Cascade: {event_type} propagates through {layers_seen[-3:]}"
                        ))
                        break
        
        return patterns
    
    def compute_confidence_multifactor(self, hypothesis: CausalHypothesis) -> float:
        """Compute confidence using multiple factors."""
        base_confidence = hypothesis.confidence
        
        # Factor 1: Evidence strength (more events = higher confidence)
        evidence_factor = min(1.0, len(hypothesis.evidence) * 0.2)
        
        # Factor 2: Causal chain length
        chain_factor = 1.0
        if hypothesis.causal_chain:
            chain_factor = min(1.0, len(hypothesis.causal_chain) * 0.15)
        
        # Factor 3: Temporal consistency
        temporal_factor = 0.8
        if hypothesis.evidence:
            steps = [e.step for e in hypothesis.evidence]
            if steps:
                temporal_factor = 1.0 - min(0.3, np.std(steps) / 100)
        
        # Combined confidence
        combined = base_confidence * 0.5 + evidence_factor * 0.2 + chain_factor * 0.15 + temporal_factor * 0.15
        return min(1.0, combined)
    
    def get_enhanced_explanations(self, failure_type: str) -> List[CausalHypothesis]:
        """Get enhanced causal explanations with multiple analysis methods."""
        hypotheses = []
        
        # Method 1: Cross-layer propagation
        hypotheses.extend(self.analyze_cross_layer_propagation())
        
        # Method 2: Temporal patterns
        patterns = self.detect_temporal_patterns()
        for pattern in patterns:
            hypotheses.append(CausalHypothesis(
                description=pattern.description,
                confidence=pattern.confidence,
                evidence=pattern.events,
                causal_chain=[pattern.pattern_type]
            ))
        
        # Enhance confidence scores
        enhanced = []
        for h in hypotheses:
            h.confidence = self.compute_confidence_multifactor(h)
            enhanced.append(h)
        
        # Sort by confidence
        enhanced.sort(key=lambda x: x.confidence, reverse=True)
        return enhanced


def enhance_with_granger_style(events: List[SemanticEvent]) -> List[CausalHypothesis]:
    """
    Entry point: Enhanced causal reasoning with Granger-style analysis.
    
    This adds:
    - Temporal precedence analysis
    - Cross-layer propagation detection  
    - Multi-factor confidence scoring
    """
    if not events:
        return []
    
    reasoner = EnhancedCausalReasoner(events)
    return reasoner.get_enhanced_explanations("general")