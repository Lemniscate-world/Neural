"""
Coupled failure detection — proprietary heuristics.
Copyright (c) 2026 NeuralDBG. All rights reserved.
"""

import math
from typing import Dict, List, Tuple, Any


class CouplingDetector:
    def __init__(self, dbg):
        self.dbg = dbg

    def detect(self, window: int = 5) -> List[Dict[str, Any]]:
        from neuraldbg import EventType

        couplings_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if len(self.dbg.events) < 2:
            return []

        sorted_events = sorted(self.dbg.events, key=lambda e: e.step)

        for i, event1 in enumerate(sorted_events):
            for event2 in sorted_events[i + 1 :]:
                step_diff = event2.step - event1.step
                if step_diff > window:
                    break

                if event1.layer_name != event2.layer_name:
                    confidence = min(event1.confidence, event2.confidence)
                    if (
                        event1.event_type == EventType.ACTIVATION_REGIME_SHIFT
                        and event2.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                    ):
                        confidence = min(confidence + 0.2, 1.0)

                    trigger = event1.id
                    consequence = event2.id
                    candidate = {
                        "trigger": trigger,
                        "consequence": consequence,
                        "trigger_label": f"{event1.event_type.value} in {event1.layer_name}",
                        "consequence_label": f"{event2.event_type.value} in {event2.layer_name}",
                        "step_difference": step_diff,
                        "confidence": confidence,
                        "is_causal_candidate": True,
                    }
                    key = (candidate["trigger_label"], candidate["consequence_label"])
                    existing = couplings_by_pair.get(key)
                    if (
                        existing is None
                        or candidate["confidence"] > existing["confidence"]
                        or (
                            math.isclose(
                                candidate["confidence"],
                                existing["confidence"],
                                rel_tol=1e-6,
                                abs_tol=1e-9,
                            )
                            and candidate["step_difference"]
                            < existing["step_difference"]
                        )
                    ):
                        couplings_by_pair[key] = candidate

        return list(couplings_by_pair.values())
