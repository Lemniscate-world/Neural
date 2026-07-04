"""
Causal explanation engine — proprietary reasoning logic.
Copyright (c) 2026 NeuralDBG. All rights reserved.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


class Explanator:
    def __init__(self, dbg):
        self.dbg = dbg

    def explain(self, failure_type: str = "vanishing_gradients"):
        from neuraldbg import CausalHypothesis

        hypotheses = []
        root_causes = self.get_root_causes()
        hypotheses.extend(root_causes)

        if failure_type == "vanishing_gradients":
            hypotheses.extend(self._explain_vanishing_gradients())
        elif failure_type == "exploding_gradients":
            hypotheses.extend(self._explain_exploding_gradients())
        elif failure_type == "dead_neurons":
            hypotheses.extend(self._explain_dead_neurons())
        elif failure_type == "saturated_activations":
            hypotheses.extend(self._explain_saturated_activations())
        elif failure_type == "optimizer_instability":
            hypotheses.extend(self._explain_optimizer_instability())
        elif failure_type == "data_anomaly":
            hypotheses.extend(self._explain_data_anomaly())

        # FIX-001: always check silent_loss and composite_blind_spot
        # regardless of failure_type — these are architectural issues
        # that apply to any failure mode involving composite modules.
        hypotheses.extend(self._explain_silent_loss())
        hypotheses.extend(self._explain_composite_blind_spot())

        seen = set()
        unique_hypotheses = []
        for h in hypotheses:
            if h.description not in seen:
                unique_hypotheses.append(h)
                seen.add(h.description)

        unique_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return unique_hypotheses

    def get_root_causes(self):
        from neuraldbg import CausalHypothesis

        hypotheses = []
        for failure_key, layer_name in self.dbg.first_failure_layer.items():
            step = self.dbg.first_failure_step[failure_key]
            matching_events = [
                e
                for e in self.dbg.events
                if e.layer_name == layer_name
                and e.step == step
                and self.event_matches_failure_key(e, failure_key)
            ]
            if not matching_events:
                matching_events = [
                    e
                    for e in self.dbg.events
                    if e.layer_name == layer_name and e.step == step
                ]
            evidence = matching_events[:1]
            hypotheses.append(
                CausalHypothesis(
                    description=f"Root cause candidate: {failure_key.replace('_', ' ')} originated in '{layer_name}' at step {step}",
                    confidence=0.95,
                    evidence=evidence,
                    causal_chain=[
                        f"First instance of {failure_key} detected in layer {layer_name}"
                    ],
                )
            )
        return hypotheses

    def event_matches_failure_key(self, event, failure_key: str) -> bool:
        from neuraldbg import EventType

        if "_" not in failure_key:
            return False
        domain, state = failure_key.split("_", 1)
        if domain == "gradient":
            return (
                event.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                and event.to_state == state
            )
        if domain == "activation":
            return (
                event.event_type == EventType.ACTIVATION_REGIME_SHIFT
                and event.to_state == state
            )
        if domain == "optimizer":
            return (
                event.event_type == EventType.OPTIMIZER_INSTABILITY
                and event.to_state == state
            )
        if domain == "data":
            return (
                event.event_type == EventType.DATA_ANOMALY and event.to_state == state
            )
        return False

    def get_causal_hypotheses(self):
        return self.explain()

    def trace_causal_chain(self, event_type: str) -> List[str]:
        relevant_events = [
            e for e in self.dbg.events if e.event_type.value == event_type
        ]
        return [
            f"{e.layer_name} at step {e.step}: {e.metadata}" for e in relevant_events
        ]

    def _explain_exploding_gradients(self):
        from neuraldbg import GradientHealth, EventType, CausalHypothesis

        hypotheses = []
        exploding_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
            and e.to_state == GradientHealth.EXPLODING.value
        ]
        if not exploding_events:
            return hypotheses
        first_exploding = min(exploding_events, key=lambda e: e.step)
        hypotheses.append(
            CausalHypothesis(
                description=f"Gradient explosion originated in layer '{first_exploding.layer_name}' at step {first_exploding.step}",
                confidence=first_exploding.confidence,
                evidence=[first_exploding],
                causal_chain=[f"Explosion detected in {first_exploding.layer_name}"],
            )
        )
        return hypotheses

    def _explain_dead_neurons(self):
        from neuraldbg import ActivationHealth, EventType, CausalHypothesis

        hypotheses = []
        dead_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.ACTIVATION_REGIME_SHIFT
            and e.to_state == ActivationHealth.DEAD.value
        ]
        if not dead_events:
            return hypotheses
        first_dead = min(dead_events, key=lambda e: e.step)
        hypotheses.append(
            CausalHypothesis(
                description=f"Neuron death detected in layer '{first_dead.layer_name}' at step {first_dead.step}",
                confidence=first_dead.confidence,
                evidence=[first_dead],
                causal_chain=[
                    f"High dead_ratio ({first_dead.metadata.get('current_dead', 1.0):.2f}) in {first_dead.layer_name}"
                ],
            )
        )
        return hypotheses

    def _explain_saturated_activations(self):
        from neuraldbg import ActivationHealth, EventType, CausalHypothesis

        hypotheses = []
        sat_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.ACTIVATION_REGIME_SHIFT
            and e.to_state == ActivationHealth.SATURATED.value
        ]
        if not sat_events:
            return hypotheses
        first_sat = min(sat_events, key=lambda e: e.step)
        hypotheses.append(
            CausalHypothesis(
                description=f"Activation saturation detected in layer '{first_sat.layer_name}' at step {first_sat.step}",
                confidence=first_sat.confidence,
                evidence=[first_sat],
                causal_chain=[
                    "High saturation_ratio "
                    f"({first_sat.metadata.get('current_saturation', first_sat.metadata.get('saturation_ratio', 1.0)):.2f}) "
                    f"in {first_sat.layer_name}"
                ],
            )
        )
        return hypotheses

    def _explain_vanishing_gradients(self):
        from neuraldbg import (
            GradientHealth,
            ActivationHealth,
            EventType,
            CausalHypothesis,
        )

        hypotheses = []
        vanishing_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
            and e.to_state == GradientHealth.VANISHING.value
        ]
        if not vanishing_events:
            return hypotheses
        first_vanishing = min(vanishing_events, key=lambda e: e.step)
        hypotheses.append(
            CausalHypothesis(
                description=f"Gradient vanishing originated in layer '{first_vanishing.layer_name}' at step {first_vanishing.step}",
                confidence=first_vanishing.confidence,
                evidence=[first_vanishing],
                causal_chain=[f"Vanishing detected in {first_vanishing.layer_name}"],
            )
        )
        saturation_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.ACTIVATION_REGIME_SHIFT
            and e.to_state == ActivationHealth.SATURATED.value
            and e.step <= first_vanishing.step + 10
        ]
        if saturation_events:
            nearest_sat = min(
                saturation_events, key=lambda e: abs(e.step - first_vanishing.step)
            )
            hypotheses.append(
                CausalHypothesis(
                    description=f"Gradient vanishing likely due to LR x activation mismatch - saturation in '{nearest_sat.layer_name}' preceded vanishing",
                    confidence=min(first_vanishing.confidence, nearest_sat.confidence),
                    evidence=[first_vanishing, nearest_sat],
                    causal_chain=[
                        f"Saturation in {nearest_sat.layer_name} at step {nearest_sat.step}",
                        f"Led to vanishing gradients in {first_vanishing.layer_name} at step {first_vanishing.step}",
                    ],
                )
            )
        return hypotheses

    def _explain_optimizer_instability(self):
        from neuraldbg import (
            OptimizerHealth,
            GradientHealth,
            EventType,
            CausalHypothesis,
        )

        hypotheses = []
        loss_spikes = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.OPTIMIZER_INSTABILITY
            and e.to_state == OptimizerHealth.LOSS_SPIKE.value
        ]
        loss_plateaus = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.OPTIMIZER_INSTABILITY
            and e.to_state == OptimizerHealth.LOSS_PLATEAU.value
        ]
        diverging = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.OPTIMIZER_INSTABILITY
            and e.to_state == OptimizerHealth.DIVERGING.value
        ]

        if loss_spikes:
            first = min(loss_spikes, key=lambda e: e.step)
            last = max(loss_spikes, key=lambda e: e.step)
            description = (
                f"Loss spike detected at step {first.step}. "
                f"Loss jumped from {first.metadata.get('recent_losses', ['?'])[-2] if len(first.metadata.get('recent_losses', [])) > 1 else '?'} "
                f"to {first.metadata.get('current_loss', '?')}. "
                "Check for gradient explosion, corrupted batch, or extremely high learning rate."
            )
            hypotheses.append(
                CausalHypothesis(
                    description=description,
                    confidence=0.85,
                    evidence=[first],
                    causal_chain=[f"Loss spike at step {first.step}"],
                )
            )

        if diverging:
            first = min(diverging, key=lambda e: e.step)
            hypotheses.append(
                CausalHypothesis(
                    description=f"Training divergence (NaN/Inf loss) detected at step {first.step}. "
                    "Check for numerical instability, learning rate explosion, or corrupted data.",
                    confidence=1.0,
                    evidence=diverging,
                    causal_chain=[f"Divergence at step {first.step}"],
                )
            )

        if loss_plateaus:
            first = min(loss_plateaus, key=lambda e: e.step)
            hypotheses.append(
                CausalHypothesis(
                    description=f"Loss plateau detected at step {first.step}. "
                    "Training may be stuck in a local minimum or the learning rate "
                    "is too small to make progress.",
                    confidence=0.75,
                    evidence=[first],
                    causal_chain=[f"Plateau at {first.step}"],
                )
            )

        # Cross-reference: if gradient explosion preceded the spike/divergence
        first_event = None
        if loss_spikes:
            first_event = min(loss_spikes, key=lambda e: e.step)
        if diverging and (
            first_event is None
            or min(diverging, key=lambda e: e.step).step < first_event.step
        ):
            first_event = min(diverging, key=lambda e: e.step)
        if first_event and first_event.to_state in (
            OptimizerHealth.LOSS_SPIKE.value,
            OptimizerHealth.DIVERGING.value,
        ):
            exploding_before = [
                e
                for e in self.dbg.events
                if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                and e.to_state == GradientHealth.EXPLODING.value
                and e.step <= first_event.step
            ]
            if exploding_before:
                grad_event = max(exploding_before, key=lambda e: e.step)
                hypotheses.append(
                    CausalHypothesis(
                        description=(
                            f"Loss {first_event.to_state} at step "
                            f"{first_event.step} was likely caused by gradient "
                            f"explosion in '{grad_event.layer_name}' at step "
                            f"{grad_event.step}."
                        ),
                        confidence=min(first_event.confidence + 0.1, 1.0),
                        evidence=[first_event, grad_event],
                        causal_chain=[
                            f"Gradient explosion in {grad_event.layer_name} "
                            f"at step {grad_event.step}",
                            f"Led to {first_event.to_state} at step {first_event.step}",
                        ],
                    )
                )

        return hypotheses

    def _explain_data_anomaly(self):
        from neuraldbg import DataHealth, EventType, CausalHypothesis

        hypotheses = []
        nan_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.DATA_ANOMALY
            and e.to_state == DataHealth.NAN_DETECTED.value
        ]
        inf_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.DATA_ANOMALY
            and e.to_state == DataHealth.INF_DETECTED.value
        ]
        shift_events = [
            e
            for e in self.dbg.events
            if e.event_type == EventType.DATA_ANOMALY
            and e.to_state == DataHealth.DISTRIBUTION_SHIFT.value
        ]

        if nan_events:
            first = min(nan_events, key=lambda e: e.step)
            hypotheses.append(
                CausalHypothesis(
                    description=f"NaN detected in '{first.layer_name}' at step {first.step} "
                    f"({first.metadata.get('nan_count', '?')} values)",
                    confidence=1.0,
                    evidence=[first],
                    causal_chain=[f"NaN spike in {first.layer_name}"],
                )
            )

        if inf_events:
            first = min(inf_events, key=lambda e: e.step)
            hypotheses.append(
                CausalHypothesis(
                    description=f"Inf detected in '{first.layer_name}' at step {first.step} "
                    f"({first.metadata.get('inf_count', '?')} values)",
                    confidence=1.0,
                    evidence=[first],
                    causal_chain=[f"Inf in {first.layer_name}"],
                )
            )

        if shift_events:
            first = min(shift_events, key=lambda e: e.step)
            metadata = first.metadata
            hypotheses.append(
                CausalHypothesis(
                    description=f"Distribution shift in '{first.layer_name}' at step {first.step}: "
                    f"mean shift {metadata.get('mean_shift_sigma', 0):.1f}σ, "
                    f"std ratio {metadata.get('current_std', 0) / max(metadata.get('prev_std', 1), 1e-9):.2f}",
                    confidence=0.85,
                    evidence=[first],
                    causal_chain=[f"Shift in {first.layer_name}"],
                )
            )

        return hypotheses

    def _explain_silent_loss(self):
        """FIX-001: surface a causal hypothesis when training produced zero
        gradient_health_transition events despite running >= 3 steps.

        This is the failure mode revealed by pytorch/pytorch#41508
        (NaN gradients in nn.MultiheadAttention): the model looks
        healthy on the surface (loss prints, optimizer steps) but
        parameters are silently corrupted because the architecture is
        fully composite and the auto leaf-only hooks are blind.

        The hypothesis is reported even when no root cause was found
        via get_root_causes(), so it is wired into the explain() output
        regardless of the user's chosen failure_type.
        """
        from neuraldbg import EventType, CausalHypothesis

        hypotheses = []
        grad_event_count = sum(
            1
            for e in self.dbg.events
            if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
        )
        step = getattr(self.dbg, "step", 0)
        warning_emitted = getattr(self.dbg, "_silent_loss_warning_emitted", False)

        n_composite = len(getattr(self.dbg, "_composite_modules", []))
        # Trigger when: steps executed AND no gradient events on internal
        # params (composite modules present but no gradient health transitions).
        # Also trigger when composite hooks were registered but gradient
        # events are still missing (the hook didn't help — wrong module?).
        should_trigger = (step >= 2 and n_composite > 0 and grad_event_count == 0) or (
            step >= 2 and n_composite > 0 and grad_event_count > 0 and warning_emitted
        )
        if should_trigger:
            confidence = 0.95 if warning_emitted else 0.80
            composite_note = (
                f" The user has registered {n_composite} composite hook(s) which "
                "should have surfaced gradient events. Check that the wrapped "
                "module is the one actually being optimized."
                if n_composite > 0
                else " No composite hooks were registered; the model is likely "
                "fully composite (e.g. nn.MultiheadAttention, custom fused "
                "kernels). Call dbg.register_composite_hook(module) to opt-in."
            )
            hypotheses.append(
                CausalHypothesis(
                    description=(
                        f"Silent loss: {step} training step(s) executed but no "
                        "gradient_health_transition event was captured. "
                        "Training may be silently corrupting parameters on a "
                        f"composite module (BUG-001).{composite_note}"
                    ),
                    confidence=confidence,
                    evidence=[],
                    causal_chain=[
                        f"{step} steps recorded with zero gradient events",
                        "Model is likely fully composite (no internal leaf modules)",
                        "FIX-001: call dbg.register_composite_hook(module) to "
                        "opt-in to instrumenting the composite module",
                    ],
                )
            )
        return hypotheses

    def _explain_composite_blind_spot(self):
        """FIX-001: surface a causal hypothesis when the wrapped model has
        no internal leaf modules (i.e. all leaves the auto installer
        could attach to are the model root itself).

        Common cause: the architecture is fully composite
        (nn.MultiheadAttention, a custom fused block, a custom autograd
        Function). Internal parameters (e.g. MHA's in_proj_weight) are
        blind to NeuralDBG unless the user opts in via
        register_composite_hook().
        """
        from neuraldbg import CausalHypothesis

        hypotheses = []
        hooked_leaf_count = getattr(self.dbg, "_hooked_leaf_count", None)
        n_composite = len(getattr(self.dbg, "_composite_modules", []))
        model = getattr(self.dbg, "model", None)

        if hooked_leaf_count is None:
            return hypotheses

        # Trigger when: no composite hooks registered AND few leaves
        # (model is fully composite and blind to auto-hooks).
        # OR: composite hooks registered but still blind (wrong module).
        should_trigger = (
            hooked_leaf_count <= 1 and n_composite == 0 and model is not None
        ) or (n_composite > 0 and hooked_leaf_count <= 2 and model is not None)
        if should_trigger:
            class_name = type(model).__name__
            hypotheses.append(
                CausalHypothesis(
                    description=(
                        f"Composite-module blind spot on '{class_name}': the model "
                        "exposes no internal leaf modules. Auto-installed "
                        "forward/backward hooks will NOT see internal parameters "
                        "(BUG-001 / pytorch/pytorch#41508). Call "
                        "dbg.register_composite_hook(model) to opt-in."
                    ),
                    confidence=0.90,
                    evidence=[],
                    causal_chain=[
                        f"_hooked_leaf_count = {hooked_leaf_count} (only the root)",
                        f"type(model) = {class_name} is fully composite",
                        "FIX-001: register_composite_hook() bypasses the leaf "
                        "filter and re-uses the same hook pair",
                    ],
                )
            )
        return hypotheses

    def collapse_events(self):
        from neuraldbg import EventType, SemanticEvent

        if not self.dbg.events:
            return []

        baseline_events = []
        transition_events = []
        for event in self.dbg.events:
            if event.from_state == "NONE":
                baseline_events.append(event)
            else:
                transition_events.append(event)

        groups: Dict[Tuple[str, str], List[SemanticEvent]] = {}
        for event in transition_events:
            key = (event.layer_name, event.event_type.value)
            if key not in groups:
                groups[key] = []
            groups[key].append(event)

        collapsed = list(baseline_events)
        for key, group in groups.items():
            sorted_group = sorted(group, key=lambda e: e.step)
            if len(sorted_group) <= 1:
                collapsed.extend(sorted_group)
                continue

            has_reversion = False
            seen_to_states = set()
            for event in sorted_group:
                if event.to_state in seen_to_states:
                    has_reversion = True
                    break
                seen_to_states.add(event.from_state)

            if has_reversion:
                collapsed.extend(sorted_group)
            else:
                first = sorted_group[0]
                last = sorted_group[-1]
                merged_metadata = dict(first.metadata)
                merged_metadata["collapsed_count"] = len(sorted_group)
                merged_metadata["step_range"] = f"{first.step}-{last.step}"
                collapsed.append(
                    SemanticEvent(
                        event_type=first.event_type,
                        layer_name=first.layer_name,
                        step=first.step,
                        from_state=first.from_state,
                        to_state=last.to_state,
                        confidence=max(e.confidence for e in sorted_group),
                        metadata=merged_metadata,
                    )
                )

        collapsed.sort(key=lambda e: e.step)
        return collapsed

    def export_aquarium_package(self, package_path: str) -> str:
        import json
        from pathlib import Path

        path = Path(package_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        package = {
            "version": "1.0",
            "model_type": "unspecified",
            "events": [self._event_to_dict(e) for e in self.dbg.events],
            "hypotheses": [self._hypothesis_to_dict(h) for h in self.explain()],
            "couplings": self.dbg.detect_coupled_failures(),
            "first_failure_layer": self.dbg.first_failure_layer,
            "first_failure_step": self.dbg.first_failure_step,
            "loss_history": self.dbg.loss_history,
        }
        path.write_text(json.dumps(package, indent=2, default=str))
        return str(path)

    def _event_to_dict(self, event):
        return {
            "id": getattr(event, "id", None),
            "type": event.event_type.value,
            "layer": event.layer_name,
            "step": event.step,
            "from": event.from_state,
            "to": event.to_state,
            "confidence": event.confidence,
            "metadata": event.metadata,
        }

    def _hypothesis_to_dict(self, hypothesis):
        return {
            "description": hypothesis.description,
            "confidence": hypothesis.confidence,
            "evidence": [self._event_to_dict(e) for e in hypothesis.evidence],
            "causal_chain": hypothesis.causal_chain,
        }

    def export_mermaid_causal_graph(self) -> str:
        lines = ["graph TD"]

        for event in self.dbg.events:
            label = (
                f"{event.event_type.value} in {event.layer_name} (Step {event.step})"
            )
            lines.append(f'    E_{event.id}["{label}"]')

        couplings = self.dbg.detect_coupled_failures()
        for coupling in couplings:
            lines.append(
                f"    E_{coupling['trigger']} -->|coupled| E_{coupling['consequence']}"
            )

        layer_events: Dict[str, List[str]] = {}
        for event in self.dbg.events:
            if event.layer_name not in layer_events:
                layer_events[event.layer_name] = []
            layer_events[event.layer_name].append(event.id)

        for layer, ids in layer_events.items():
            for j in range(len(ids) - 1):
                lines.append(f"    E_{ids[j]} -->|temporal| E_{ids[j + 1]}")

        return "\n".join(lines)
