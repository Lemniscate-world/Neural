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

    def check_anomaly(self, tensor: torch.Tensor, layer_name: str, stats_key: str = None):
        from neuraldbg import DataHealth, SemanticEvent, EventType

        stats_key = stats_key or layer_name
        current_health, health_metadata = self.classify_health(tensor)

        if current_health not in (DataHealth.NAN_DETECTED, DataHealth.INF_DETECTED):
            t = tensor.detach().float()
            current_mean = t.mean().item() if t.numel() > 0 else 0.0
            current_std = t.std().item() if t.numel() > 1 else 0.0

            if stats_key in self.dbg.previous_input_stats:
                prev = self.dbg.previous_input_stats[stats_key]
                prev_std = prev.get("std", 1.0)
                if prev_std > 1e-9:
                    mean_shift = abs(current_mean - prev.get("mean", 0.0)) / prev_std
                    std_ratio = current_std / prev_std if prev_std > 0 else 1.0
                    # Calibrated thresholds: family-aware + strict mode
                    fm = getattr(self.dbg, '_family_mult', 1.0)
                    sm = 2.0 if getattr(self.dbg, 'strict_mode', False) else 1.0
                    if (mean_shift > 4.0 * fm * sm
                            or std_ratio > 8.0 * fm * sm
                            or std_ratio < 0.1 / (fm * sm)):
                        current_health = DataHealth.DISTRIBUTION_SHIFT
                        health_metadata = {
                            "prev_mean": prev.get("mean", 0.0),
                            "current_mean": current_mean,
                            "prev_std": prev_std,
                            "current_std": current_std,
                            "mean_shift_sigma": mean_shift,
                        }

            self.dbg.previous_input_stats[stats_key] = {
                "mean": current_mean,
                "std": current_std,
            }

        prev_health = self.dbg.previous_data_health.get(layer_name, DataHealth.NORMAL)

        # Anti-oscillation debounce + per-step gate.
        # The streak advances only across DISTINCT training steps, so the
        # two checks per step (module input + output) cannot inflate it.
        # NaN/INF are hard errors and bypass the debounce entirely.
        if not hasattr(self.dbg, '_data_health_streak'):
            self.dbg._data_health_streak = {}
        if not hasattr(self.dbg, '_data_emitted_this_step'):
            self.dbg._data_emitted_this_step = set()
        current_step = self.dbg.step
        if not hasattr(self, '_last_data_step'):
            self._last_data_step = current_step
        if current_step != self._last_data_step:
            self.dbg._data_emitted_this_step.clear()
            self._last_data_step = current_step

        hard_anomaly = current_health in (DataHealth.NAN_DETECTED, DataHealth.INF_DETECTED)
        layer_streak = self.dbg._data_health_streak.get(layer_name)
        if hard_anomaly:
            new_streak = 2
        elif layer_streak and layer_streak[0] == current_health.value and layer_streak[1] == current_step - 1:
            new_streak = layer_streak[2] + 1
        elif layer_streak and layer_streak[0] == current_health.value and layer_streak[1] == current_step:
            new_streak = layer_streak[2]
        else:
            new_streak = 1
        self.dbg._data_health_streak[layer_name] = (current_health.value, current_step, new_streak)
        should_emit = (current_health != prev_health) and (new_streak >= 2)
        already_emitted = layer_name in self.dbg._data_emitted_this_step

        if should_emit and not already_emitted:
            # Suppress classifier-head distribution_shift noise in non-strict mode
            is_classifier_noise = (
                current_health == DataHealth.DISTRIBUTION_SHIFT
                and not getattr(self.dbg, 'strict_mode', False)
                and any(kw in layer_name.lower() for kw in ('fc', 'head', 'classifier'))
            )
            if not is_classifier_noise:
                if current_health != DataHealth.NORMAL:
                    self.dbg._track_first_occurrence(
                        f"data_{current_health.value}", layer_name
                    )
                    if current_health in (DataHealth.NAN_DETECTED, DataHealth.INF_DETECTED, DataHealth.DISTRIBUTION_SHIFT):
                        if hasattr(self.dbg, "disk_cache"):
                            cache_path = self.dbg.disk_cache.save(tensor, prefix=f"anomaly_{layer_name}")
                            health_metadata["tensor_cache_path"] = cache_path

                confidence = 1.0
                if current_health == DataHealth.DISTRIBUTION_SHIFT:
                    mean_shift_val = health_metadata.get("mean_shift_sigma", 3.0)
                    confidence = min(mean_shift_val * 0.2, 1.0)

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
                self.dbg._data_emitted_this_step.add(layer_name)
            self.dbg.previous_data_health[layer_name] = current_health
