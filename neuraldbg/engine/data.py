"""
Data anomaly detection — proprietary heuristics.
Copyright (c) 2026 NeuralDBG. All rights reserved.
"""

import torch
from typing import Dict, Tuple


class DataAnalyzer:
    def __init__(self, dbg):
        self.dbg = dbg

    def classify_health(self, tensor: torch.Tensor):
        from neuraldbg import DataHealth

        t = tensor.detach().float()

        has_nan = torch.isnan(t).any().item()
        if has_nan:
            return DataHealth.NAN_DETECTED, {
                "nan_count": int(torch.isnan(t).sum().item())
            }

        has_inf = torch.isinf(t).any().item()
        if has_inf:
            return DataHealth.INF_DETECTED, {
                "inf_count": int(torch.isinf(t).sum().item())
            }

        return DataHealth.NORMAL, {}

    def check_anomaly(self, tensor: torch.Tensor, layer_name: str):
        from neuraldbg import DataHealth, SemanticEvent, EventType

        current_health, health_metadata = self.classify_health(tensor)

        if current_health not in (DataHealth.NAN_DETECTED, DataHealth.INF_DETECTED):
            t = tensor.detach().float()
            current_mean = t.mean().item()
            current_std = t.std().item()

            if layer_name in self.dbg.previous_input_stats:
                prev = self.dbg.previous_input_stats[layer_name]
                prev_std = prev.get("std", 1.0)
                if prev_std > 1e-9:
                    mean_shift = abs(current_mean - prev.get("mean", 0.0)) / prev_std
                    std_ratio = current_std / prev_std if prev_std > 0 else 1.0
                    if mean_shift > 3.0 or std_ratio > 5.0 or std_ratio < 0.2:
                        current_health = DataHealth.DISTRIBUTION_SHIFT
                        health_metadata = {
                            "prev_mean": prev.get("mean", 0.0),
                            "current_mean": current_mean,
                            "prev_std": prev_std,
                            "current_std": current_std,
                            "mean_shift_sigma": mean_shift,
                        }

            self.dbg.previous_input_stats[layer_name] = {
                "mean": current_mean,
                "std": current_std,
            }

        prev_health = self.dbg.previous_data_health.get(layer_name, DataHealth.NORMAL)

        if current_health != prev_health:
            if current_health != DataHealth.NORMAL:
                self.dbg._track_first_occurrence(
                    f"data_{current_health.value}", layer_name
                )
                # Save anomalous tensor to disk to prevent VRAM/RAM OOMs
                if current_health in (DataHealth.NAN_DETECTED, DataHealth.INF_DETECTED, DataHealth.DISTRIBUTION_SHIFT):
                    if hasattr(self.dbg, "disk_cache"):
                        cache_path = self.dbg.disk_cache.save(tensor, prefix=f"anomaly_{layer_name}")
                        health_metadata["tensor_cache_path"] = cache_path

            confidence = 1.0
            if current_health == DataHealth.DISTRIBUTION_SHIFT:
                mean_shift_val = health_metadata.get("mean_shift_sigma", 3.0)
                confidence = min(mean_shift_val * 0.2, 1.0)

            # Sample resources once per step
            resource_snapshot, resource_baseline = self.dbg._get_step_resource_snapshot(tensor.device)
            is_spike, spike_keys = self.dbg._is_memory_spike(resource_snapshot, resource_baseline)
            health_metadata["resources"] = resource_snapshot
            health_metadata["memory_spike"] = is_spike
            health_metadata["memory_spike_keys"] = spike_keys

            self.dbg.events.append(
                SemanticEvent(
                    event_type=EventType.DATA_ANOMALY,
                    layer_name=layer_name,
                    step=self.dbg.step,
                    from_state=prev_health.value,
                    to_state=current_health.value,
                    confidence=confidence,
                    metadata=health_metadata,
                )
            )
            self.dbg.previous_data_health[layer_name] = current_health
