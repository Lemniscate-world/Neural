"""
Activation health analysis — proprietary heuristics.
Copyright (c) 2026 NeuralDBG. All rights reserved.
"""

from typing import Dict, Optional, Any


class ActivationAnalyzer:
    def __init__(self, dbg):
        self.dbg = dbg

    def classify_health(self, stats: Dict[str, float]):
        from neuraldbg import ActivationHealth

        dead_ratio = stats.get("dead_ratio", 0.0)
        sat_ratio = stats.get("saturation_ratio", 0.0)
        has_nan = stats.get("has_nan", False)
        has_inf = stats.get("has_inf", False)

        if has_nan or has_inf:
            return ActivationHealth.ANOMALOUS
        if dead_ratio > 0.9:
            return ActivationHealth.DEAD
        if sat_ratio > 0.5:
            return ActivationHealth.SATURATED
        return ActivationHealth.NORMAL

    def detect_shift(
        self, prev_stats: Dict[str, float], current_stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        from neuraldbg import ActivationHealth

        prev_health = self.classify_health(prev_stats)
        curr_health = self.classify_health(current_stats)
        if prev_health != curr_health:
            return {
                "type": f"{prev_health.value}_to_{curr_health.value}",
                "confidence": 0.9,
            }
        return None
