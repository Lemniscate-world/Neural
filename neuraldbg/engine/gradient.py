"""
Gradient health analysis — proprietary heuristics.
Copyright (c) 2026 NeuralDBG. All rights reserved.
"""

from typing import Dict, Optional, Any


class GradientAnalyzer:
    def __init__(self, dbg):
        self.dbg = dbg

    def classify_health(self, norm: float):
        from neuraldbg import GradientHealth

        if norm < self.dbg.threshold_vanishing:
            return GradientHealth.VANISHING
        elif norm > self.dbg.threshold_exploding:
            return GradientHealth.EXPLODING
        else:
            # P2b: the former SATURATED band [threshold_vanishing,
            # threshold_vanishing * 100) mislabelled healthy small gradients
            # as saturated (saturation is an activation-regime concept)
            return GradientHealth.HEALTHY

    def detect_transition(
        self, prev_norm: float, current_norm: float
    ) -> Optional[Dict[str, Any]]:
        from neuraldbg import GradientHealth

        prev_health = self.classify_health(prev_norm)
        current_health = self.classify_health(current_norm)

        if prev_health != current_health:
            if prev_norm > 0:
                ratio = abs(current_norm - prev_norm) / prev_norm
            else:
                ratio = abs(current_norm)
            confidence = min(ratio * 0.1, 1.0)
            return {
                "type": f"{prev_health.value}_to_{current_health.value}",
                "confidence": confidence,
            }
        return None
