"""Causal Chain Engine — true causal inference on computation graphs.

Transforms flat NeuralDBG events into directed causal chains:
  Event A (step 5, layer_3) -> Event B (step 8, layer_2) -> Event C (step 12, loss)

Rules:
  R1 (Backprop): Anomaly at layer L, step S -> may cause anomaly at layer L-1, step S+1
  R2 (Temporal): Dead neuron at step S -> may cause vanishing gradient at step S+delta
  R3 (Optimizer): Instability at step S -> may cause explosion at step S+1
  R4 (Data): Data anomaly at step S -> may cause any anomaly at any layer at step S+1

Each chain has a confidence score based on:
  - Temporal proximity (closer = stronger)
  - Layer adjacency (adjacent layers = stronger)
  - Event type compatibility (matching rules = stronger)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CausalLink:
    """A directed edge in the causal graph."""
    source_event: Dict
    target_event: Dict
    rule: str  # which rule created this link
    confidence: float
    evidence: str


@dataclass
class CausalChain:
    """A complete causal chain from root cause to final symptom."""
    links: List[CausalLink] = field(default_factory=list)
    root_cause: Optional[str] = None
    final_symptom: Optional[str] = None
    confidence: float = 0.0
    description: str = ""

    @property
    def length(self) -> int:
        return len(self.links)

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "root_cause": self.root_cause,
            "final_symptom": self.final_symptom,
            "chain_length": self.length,
            "confidence": round(self.confidence, 3),
            "links": [
                {
                    "from": f"{l.source_event.get('layer_name','?')}@{l.source_event.get('step','?')}",
                    "to": f"{l.target_event.get('layer_name','?')}@{l.target_event.get('step','?')}",
                    "rule": l.rule,
                    "confidence": round(l.confidence, 3),
                    "evidence": l.evidence,
                }
                for l in self.links
            ],
        }


# Event type compatibility matrix for causal rules
# (source_type, target_type) -> base_confidence
CAUSAL_COMPATIBILITY = {
    # Dead neurons cause vanishing gradients
    ("activation_regime_shift", "gradient_health_transition"): 0.6,
    # Gradient explosion causes optimizer instability
    ("gradient_health_transition", "optimizer_instability"): 0.7,
    # Optimizer instability causes further gradient issues
    ("optimizer_instability", "gradient_health_transition"): 0.5,
    # Data anomaly causes anything
    ("data_anomaly", "gradient_health_transition"): 0.8,
    ("data_anomaly", "optimizer_instability"): 0.8,
    ("data_anomaly", "activation_regime_shift"): 0.6,
    # Gradient-health-only propagation is noise; removed.
    # Real causal chains start with data_anomaly, optimizer_instability,
    # or activation_regime_shift and propagate TO gradient issues.
    ("optimizer_instability", "optimizer_instability"): 0.5,
    ("silent_corruption", "gradient_health_transition"): 0.7,
    ("silent_corruption", "optimizer_instability"): 0.6,
    ("nan_detected", "gradient_health_transition"): 0.9,
    ("nan_detected", "optimizer_instability"): 0.9,
    ("nan_detected", "activation_regime_shift"): 0.7,
    # Gradient propagation: low confidence, filtered by quality gate if all healthy
    ("gradient_health_transition", "gradient_health_transition"): 0.25,
}


def _state_is_problematic(state: str) -> bool:
    """Check if a state indicates a problem (not healthy/normal)."""
    return state.lower() not in ("healthy", "none", "normal", "")


def _layer_depth(layer_name: str) -> int:
    """Estimate layer depth from name (lower = closer to input)."""
    name = layer_name.lower()
    # Layers with numbers: Linear_0, Conv2d_1, ReLU_2
    import re
    nums = re.findall(r'(\d+)', name)
    if nums:
        return int(nums[-1])
    # Root is always deepest (closest to loss)
    if "root" in name:
        return 999
    return 0


def _temporal_distance(e1: Dict, e2: Dict) -> int:
    """Steps between two events."""
    s1 = e1.get("step", 0)
    s2 = e2.get("step", 0)
    return abs(s2 - s1)


def _layer_distance(e1: Dict, e2: Dict) -> int:
    """Layer depth difference between two events."""
    return abs(_layer_depth(e1.get("layer_name", "")) - _layer_depth(e2.get("layer_name", "")))


def build_causal_graph(events: List[Dict], max_temporal_gap: int = 5) -> List[CausalLink]:
    """Build a directed causal graph from NeuralDBG events.

    Only considers events with problematic states (not healthy/normal).
    Links are created when temporal + layer proximity suggests causation.
    """
    # Filter: keep all significant events, skip only normal activation shifts
    significant = []
    for e in events:
        et = e.get("event_type", "")
        if et == "activation_regime_shift" and not _state_is_problematic(e.get("to_state", "")):
            continue  # skip normal activation transitions
        significant.append(e)

    if len(significant) < 2:
        return []

    links = []
    for i, src in enumerate(significant):
        for j, tgt in enumerate(significant):
            if i == j:
                continue
            # Only forward in time (or same step for cross-layer)
            if tgt.get("step", 0) < src.get("step", 0):
                continue

            temporal_dist = _temporal_distance(src, tgt)
            if temporal_dist > max_temporal_gap:
                continue

            # Only cross-layer or cross-type
            same_layer = src.get("layer_name") == tgt.get("layer_name")
            same_type = src.get("event_type") == tgt.get("event_type")
            if same_layer and same_type:
                continue

            # Check compatibility
            type_pair = (src.get("event_type", ""), tgt.get("event_type", ""))
            base_conf = CAUSAL_COMPATIBILITY.get(type_pair, 0.0)  # 0 = no compatibility

            # Require at least one side to be genuinely problematic
            src_problematic = _state_is_problematic(src.get("to_state", src.get("from_state", "")))
            tgt_problematic = _state_is_problematic(tgt.get("to_state", tgt.get("from_state", "")))
            if not (src_problematic or tgt_problematic):
                continue  # skip healthy-to-healthy entirely

            # Bonus if BOTH sides are problematic
            if src_problematic and tgt_problematic:
                base_conf *= 1.3

            # Adjust confidence by proximity
            temporal_bonus = max(0, 1.0 - temporal_dist / max_temporal_gap)
            layer_bonus = 1.0 if not same_layer else 0.5
            confidence = base_conf * temporal_bonus * layer_bonus

            if confidence > 0.15:  # minimum threshold
                evidence = (
                    f"{src.get('event_type')} at {src.get('layer_name')} step {src.get('step')} "
                    f"-> {tgt.get('event_type')} at {tgt.get('layer_name')} step {tgt.get('step')} "
                    f"(gap={temporal_dist}, base_conf={base_conf:.2f})"
                )
                links.append(CausalLink(
                    source_event=src,
                    target_event=tgt,
                    rule=f"Temporal({temporal_dist})",
                    confidence=min(confidence, 1.0),
                    evidence=evidence,
                ))

    # Sort by confidence
    links.sort(key=lambda l: l.confidence, reverse=True)
    return links


def extract_chains(links: List[CausalLink], min_length: int = 2, max_chains: int = 30) -> List[CausalChain]:
    """Extract causal chains from the link graph by finding connected paths."""
    if not links:
        return []

    # Build adjacency using Python object id() for unique node keys
    adjacency: Dict[str, List[Tuple[str, CausalLink]]] = {}
    for link in links:
        src_key = str(id(link.source_event))
        tgt_key = str(id(link.target_event))
        if src_key not in adjacency:
            adjacency[src_key] = []
        adjacency[src_key].append((tgt_key, link))

    # Find chains via DFS with node-based cycle detection
    chains: List[CausalChain] = []
    visited_chains = set()

    def _emit_chain(path: List[CausalLink]):
        if len(chains) >= max_chains:
            return
        # Quality gate: chain must contain at least one genuinely problematic transition
        has_problem = any(
            _state_is_problematic(l.source_event.get("to_state", l.source_event.get("from_state", "")))
            or _state_is_problematic(l.target_event.get("to_state", l.target_event.get("from_state", "")))
            for l in path
        )
        if not has_problem:
            return  # skip chains that are pure healthy propagation

        chain = CausalChain(links=list(path))
        # Better root cause: first problematic event, not just first event
        chain.root_cause = path[0].source_event.get("event_type", "?")
        chain.final_symptom = path[-1].target_event.get("event_type", "?")
        # Override root/final with actual problematic states
        for l in path:
            src_st = l.source_event.get("to_state", "")
            if _state_is_problematic(src_st):
                chain.root_cause = f"{l.source_event.get('event_type','?')}[{src_st}]"
                break
        for l in reversed(path):
            tgt_st = l.target_event.get("to_state", "")
            if _state_is_problematic(tgt_st):
                chain.final_symptom = f"{l.target_event.get('event_type','?')}[{tgt_st}]"
                break

        chain.confidence = sum(l.confidence for l in path) / len(path)
        # Compact fingerprint for dedup
        fp = tuple(
            (l.source_event.get("event_type",""), l.source_event.get("layer_name",""),
             l.target_event.get("event_type",""), l.target_event.get("layer_name",""))
            for l in path
        )
        if fp not in visited_chains:
            visited_chains.add(fp)
            # Description with state information
            parts = []
            for l in path:
                src_st = l.source_event.get("to_state", l.source_event.get("from_state", "?"))
                parts.append(f"{l.source_event.get('event_type','?')}({l.source_event.get('layer_name','?')})[{src_st}]")
            last_st = path[-1].target_event.get("to_state", path[-1].target_event.get("from_state", "?"))
            parts.append(f"{path[-1].target_event.get('event_type','?')}({path[-1].target_event.get('layer_name','?')})[{last_st}]")
            chain.description = " -> ".join(parts)
            chains.append(chain)

    def dfs(node: str, path: List[CausalLink], visited_nodes: set, depth: int):
        if depth > 6 or len(chains) >= max_chains:
            return
        # Emit chain if path is long enough
        if len(path) >= min_length:
            _emit_chain(path)
        # Continue deeper — avoid revisiting nodes
        if node in adjacency:
            for next_node, link in adjacency[node]:
                if next_node not in visited_nodes and link not in path:
                    dfs(next_node, path + [link], visited_nodes | {next_node}, depth + 1)

    # Start DFS from each source node (only best links per source to reduce fanout)
    for node in adjacency:
        if len(chains) >= max_chains:
            break
        dfs(node, [], {node}, 0)

    # Sort by: confidence * length * terminal_bonus (prefer chains ending with real problems)
    def _chain_score(c: CausalChain) -> float:
        score = c.confidence * c.length
        # Bonus for chains that end with a genuine problem (not healthy)
        if c.links:
            last = c.links[-1].target_event
            if _state_is_problematic(last.get("to_state", "")):
                score *= 1.5
            # Extra bonus for chains with diverse event types
            types_in_chain = len(set(
                l.source_event.get("event_type","") for l in c.links
            ) | {c.links[-1].target_event.get("event_type","")})
            if types_in_chain >= 3:
                score *= 1.3
        return score

    chains.sort(key=_chain_score, reverse=True)
    return chains


def explain_causal(events: List[Dict]) -> List[CausalChain]:
    """Main entry point: build causal graph and extract chains from events.

    Args:
        events: List of NeuralDBG event dicts (from dump_events()).

    Returns:
        Ranked list of causal chains, most significant first.
    """
    links = build_causal_graph(events)
    chains = extract_chains(links)
    return chains
