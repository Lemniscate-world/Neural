"""
NeuralDbg Causal Inference Engine

This module defines the NeuralDbg class, which is a causal inference engine for deep learning training dynamics.
It extracts semantic events from training, compresses them into causal patterns, and provides
post-mortem reasoning about training failures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Try to import dynamo for torch.compile suppression
try:
    import torch._dynamo as dynamo
    dynamo_disable = dynamo.disable
except ImportError:
    # Fallback for PyTorch < 2.0 or if dynamo is unavailable
    def dynamo_disable(fn):
        return fn

class EventType(Enum):
    """Types of semantic events that can occur during training."""
    GRADIENT_HEALTH_TRANSITION = "gradient_health_transition"
    ACTIVATION_REGIME_SHIFT = "activation_regime_shift"
    OPTIMIZER_INSTABILITY = "optimizer_instability"
    DATA_ANOMALY = "data_anomaly"

class GradientHealth(Enum):
    """Gradient health states."""
    HEALTHY = "healthy"
    VANISHING = "vanishing"
    EXPLODING = "exploding"
    SATURATED = "saturated"

class ActivationHealth(Enum):
    """Activation health states for semantic regime monitoring."""
    NORMAL = "normal"
    SATURATED = "saturated"
    DEAD = "dead"
    ANOMALOUS = "anomalous"

@dataclass
class SemanticEvent:
    """
    Represents a meaningful transition in training dynamics.

    Unlike raw tensor snapshots, semantic events capture high-level changes
    that are relevant for causal inference.
    """
    event_type: EventType
    layer_name: str
    step: int
    from_state: Any
    to_state: Any
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class CausalHypothesis:
    """A ranked hypothesis about the cause of a training failure."""
    description: str
    confidence: float
    evidence: List[SemanticEvent]
    causal_chain: List[str]

class NeuralDbg:
    """
    Causal inference engine for deep learning training dynamics.

    Monitors training loops to extract semantic events, detect patterns,
    and provide post-mortem explanations for training failures.
    """

    def __init__(self, model: nn.Module, threshold_vanishing: float = 1e-6, threshold_exploding: float = 1e3):
        """
        Initialize the causal inference engine.

        Args:
            model: The PyTorch model to monitor
            threshold_vanishing: Gradient norm threshold for vanishing detection
            threshold_exploding: Gradient norm threshold for exploding detection
        """
        self.model = model
        self.threshold_vanishing = threshold_vanishing
        self.threshold_exploding = threshold_exploding

        # Verify if model is already compiled
        if hasattr(torch, "_dynamo") and isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
            import warnings
            warnings.warn(
                "NeuralDbg: Model is already compiled. Hooks installed after compilation "
                "might not fire in the optimized graph. For best results, wrap the model "
                "with NeuralDbg BEFORE calling torch.compile().",
                UserWarning
            )

        # Semantic event storage (not tensors!)
        self.events: List[SemanticEvent] = []

        # Previous state tracking for transition detection
        self.previous_gradient_norms: Dict[str, float] = {}
        self.previous_activation_stats: Dict[str, Dict[str, float]] = {}
        
        # Causal tracking: First layer to fail in a specific way
        self.first_failure_step: Dict[str, int] = {} # failure_key -> step
        self.first_failure_layer: Dict[str, str] = {} # failure_key -> layer_name

        # Hook storage for automatic monitoring
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Training state
        self.step = 0
        self.is_monitoring = False

    def step_iteration(self):
        """Increment the internal step counter."""
        self.step += 1

    def get_events(self) -> List[SemanticEvent]:
        """Return all captured semantic events."""
        return self.events

    def __enter__(self):
        """Start monitoring the training loop."""
        self._install_hooks()
        self.is_monitoring = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and cleanup."""
        self._remove_hooks()
        self.is_monitoring = False

    def _install_hooks(self):
        """Install forward and backward hooks to extract semantic events."""
        @dynamo_disable
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
            """Extract semantic events from forward pass."""
            if not self.is_monitoring:
                return

            layer_name = self._get_layer_name(module)

            # Extract activation regime information
            if isinstance(output, torch.Tensor):
                activation_stats = self._compute_activation_stats(output)
                current_health = self._classify_activation_health(activation_stats)

                # Detect activation regime shifts
                if layer_name in self.previous_activation_stats:
                    prev_stats = self.previous_activation_stats[layer_name]
                    prev_health = self._classify_activation_health(prev_stats)
                    
                    if prev_health != current_health:
                        if current_health != ActivationHealth.NORMAL:
                            self._track_first_occurrence(f"activation_{current_health.value}", layer_name)
                            
                        event = SemanticEvent(
                            event_type=EventType.ACTIVATION_REGIME_SHIFT,
                            layer_name=layer_name,
                            step=self.step,
                            from_state=prev_health.value,
                            to_state=current_health.value,
                            confidence=0.9,
                            metadata={
                                'prev_saturation': prev_stats.get('saturation_ratio'),
                                'current_saturation': activation_stats.get('saturation_ratio'),
                                'prev_dead': prev_stats.get('dead_ratio'),
                                'current_dead': activation_stats.get('dead_ratio')
                            }
                        )
                        self.events.append(event)
                else:
                    # Capture baseline activation state on first encounter
                    event = SemanticEvent(
                        event_type=EventType.ACTIVATION_REGIME_SHIFT,
                        layer_name=layer_name,
                        step=self.step,
                        from_state="NONE",
                        to_state=current_health.value,
                        confidence=1.0,
                        metadata=activation_stats
                    )
                    self.events.append(event)
                    if current_health != ActivationHealth.NORMAL:
                        self._track_first_occurrence(f"activation_{current_health.value}", layer_name)

                self.previous_activation_stats[layer_name] = activation_stats

        @dynamo_disable
        def full_backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            """Extract semantic events from backward pass using full_backward_hook."""
            if not self.is_monitoring:
                return

            layer_name = self._get_layer_name(module)

            # Extract gradient health information
            # In full_backward_hook, grad_output is a tuple of gradients w.r.t. outputs
            if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                grad_norm = grad_output[0].norm().item()

                # Detect gradient health transitions
                if layer_name in self.previous_gradient_norms:
                    prev_norm = self.previous_gradient_norms[layer_name]
                    transition = self._detect_gradient_transition(prev_norm, grad_norm)
                    if transition:
                        current_health = self._classify_gradient_health(grad_norm)
                        if current_health != GradientHealth.HEALTHY:
                            self._track_first_occurrence(f"gradient_{current_health.value}", layer_name)

                        event = SemanticEvent(
                            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
                            layer_name=layer_name,
                            step=self.step,
                            from_state=self._classify_gradient_health(prev_norm).value,
                            to_state=current_health.value,
                            confidence=transition['confidence'],
                            metadata={
                                'prev_norm': prev_norm,
                                'current_norm': grad_norm,
                                'transition_type': transition['type']
                            }
                        )
                        self.events.append(event)
                else:
                    # Capture baseline gradient state on first encounter
                    current_health = self._classify_gradient_health(grad_norm)
                    event = SemanticEvent(
                        event_type=EventType.GRADIENT_HEALTH_TRANSITION,
                        layer_name=layer_name,
                        step=self.step,
                        from_state="NONE",
                        to_state=current_health.value,
                        confidence=1.0,
                        metadata={
                            'current_norm': grad_norm,
                            'transition_type': 'baseline'
                        }
                    )
                    self.events.append(event)
                    if current_health != GradientHealth.HEALTHY:
                        self._track_first_occurrence(f"gradient_{current_health.value}", layer_name)

                self.previous_gradient_norms[layer_name] = grad_norm

        # Install hooks on all modules (including root for full coverage)
        # Install hooks on leaf modules to ensure maximum compatibility with torch.compile
        for name, module in self.model.named_modules():
            # Skip root and modules with children to avoid redundant captures and root-bypass issues
            if len(list(module.children())) > 0 and name != "":
                continue
                
            self.hooks.append(module.register_forward_hook(forward_hook))
            # Use register_full_backward_hook if available (PyTorch 1.9+)
            if hasattr(module, "register_full_backward_hook"):
                self.hooks.append(module.register_full_backward_hook(full_backward_hook))
            else:
                # Fallback for older versions (though we expect >=1.9)
                self.hooks.append(module.register_backward_hook(full_backward_hook))

        # Check for DataParallel/DDP and warn/handle
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            import warnings
            warnings.warn(
                f"NeuralDbg: Model is wrapped in {type(self.model).__name__}. "
                "Hooks might not persist correctly during replication. Consider wrapping "
                "the inner module (.module) instead.",
                UserWarning
            )

    def _remove_hooks(self):
        """Remove all installed hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _get_layer_name(self, module: nn.Module) -> str:
        """Get the name of a module from the model."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name or "root"
        
        # Fallback for Dynamo-wrapped modules or internal replicas
        if hasattr(module, "_get_name"):
            return module._get_name()
        return type(module).__name__

    def _compute_activation_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistical summary of activation tensor."""
        # Ensure we are working with float32 for stats to avoid precision issues
        t_float = tensor.detach().float()
        
        # Calculate sparsity (fraction of zeros)
        # Using a small epsilon for float comparison
        sparsity = (t_float.abs() < 1e-9).float().mean().item()
        
        # Calculate dead neurons (per-neuron sparsity over batch)
        # Assuming batch is dim 0
        if t_float.dim() > 1:
            dead_ratio = (t_float.abs().sum(dim=0) < 1e-9).float().mean().item()
        else:
            dead_ratio = sparsity

        # Calculate saturation ratio (for Sigmoid or Tanh typically)
        # We consider a value saturated if it's very close to 1.0 or -1.0
        saturation_ratio = (t_float.abs() > 0.95).float().mean().item()

        return {
            'mean': t_float.mean().item(),
            'std': t_float.std().item(),
            'min': t_float.min().item(),
            'max': t_float.max().item(),
            'sparsity': sparsity,
            'dead_ratio': dead_ratio,
            'norm': t_float.norm().item(),
            'saturation_ratio': saturation_ratio
        }

    def _classify_activation_health(self, stats: Dict[str, float]) -> ActivationHealth:
        """Classify activation regime based on extracted statistics."""
        if stats.get('dead_ratio', 0) > 0.9:
            return ActivationHealth.DEAD
        elif stats.get('saturation_ratio', 0) > 0.7:
            return ActivationHealth.SATURATED
        elif stats.get('std', 1.0) < 1e-4:
            return ActivationHealth.ANOMALOUS
        else:
            return ActivationHealth.NORMAL

    def _detect_activation_shift(self, prev_stats: Dict[str, float], current_stats: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Deprecated: Use _classify_activation_health and direct transition detection instead."""
        # Kept for compatibility if needed, but preferred to use state-based transitions
        prev_health = self._classify_activation_health(prev_stats)
        curr_health = self._classify_activation_health(current_stats)
        if prev_health != curr_health:
            return {'type': f"{prev_health.value}_to_{curr_health.value}", 'confidence': 0.9}
        return None

    def _classify_gradient_health(self, norm: float) -> GradientHealth:
        """Classify gradient health based on norm."""
        if norm < self.threshold_vanishing:
            return GradientHealth.VANISHING
        elif norm > self.threshold_exploding:
            return GradientHealth.EXPLODING
        # Saturated gradients in this context refer to persistent small values 
        # that are just above vanishing but indicate diminishing flow.
        elif norm < (self.threshold_vanishing * 100):
            return GradientHealth.SATURATED
        else:
            return GradientHealth.HEALTHY

    def _track_first_occurrence(self, failure_type: str, layer_name: str):
        """Track the first layer that encountered a specific failure."""
        if failure_type not in self.first_failure_step:
            self.first_failure_step[failure_type] = self.step
            self.first_failure_layer[failure_type] = layer_name

    def _detect_gradient_transition(self, prev_norm: float, current_norm: float) -> Optional[Dict[str, Any]]:
        """Detect transitions in gradient health."""
        prev_health = self._classify_gradient_health(prev_norm)
        current_health = self._classify_gradient_health(current_norm)

        if prev_health != current_health:
            # Calculate confidence based on magnitude of change
            if prev_norm > 0:
                ratio = abs(current_norm - prev_norm) / prev_norm
            else:
                ratio = abs(current_norm)  # Handle zero case

            confidence = min(ratio * 0.1, 1.0)  # Scale down the confidence

            return {
                'type': f"{prev_health.value}_to_{current_health.value}",
                'confidence': confidence
            }
        return None

    def explain_failure(self, failure_type: str = "vanishing_gradients") -> List[CausalHypothesis]:
        """
        Provide ranked causal hypotheses for a training failure.

        Args:
            failure_type: Type of failure to explain

        Returns:
            List of ranked hypotheses with confidence scores
        """
        hypotheses = []

        # Start with root causes from first-occurrence tracking
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

        # Filter out duplicates (based on description)
        seen = set()
        unique_hypotheses = []
        for h in hypotheses:
            if h.description not in seen:
                unique_hypotheses.append(h)
                seen.add(h.description)

        # Sort by confidence
        unique_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return unique_hypotheses

    def _explain_exploding_gradients(self) -> List[CausalHypothesis]:
        """Generate hypotheses for exploding gradient failures."""
        hypotheses = []

        # Find first exploding gradient event
        exploding_events = [e for e in self.events
                          if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                          and e.to_state == GradientHealth.EXPLODING.value]

        if not exploding_events:
            return hypotheses

        first_exploding = min(exploding_events, key=lambda e: e.step)

        # Hypothesis 1: Originated in this layer
        hypotheses.append(CausalHypothesis(
            description=f"Gradient explosion originated in layer '{first_exploding.layer_name}' at step {first_exploding.step}",
            confidence=first_exploding.confidence,
            evidence=[first_exploding],
            causal_chain=[f"Explosion detected in {first_exploding.layer_name}"]
        ))

        return hypotheses

    def _explain_dead_neurons(self) -> List[CausalHypothesis]:
        """Generate hypotheses for dead neuron failures."""
        hypotheses = []

        # Use ACTIVATION_REGIME_SHIFT to detect DEAD state
        dead_events = [e for e in self.events
                      if e.event_type == EventType.ACTIVATION_REGIME_SHIFT
                      and e.to_state == ActivationHealth.DEAD.value]

        if not dead_events:
            return hypotheses

        first_dead = min(dead_events, key=lambda e: e.step)

        hypotheses.append(CausalHypothesis(
            description=f"Neuron death detected in layer '{first_dead.layer_name}' at step {first_dead.step}",
            confidence=first_dead.confidence,
            evidence=[first_dead],
            causal_chain=[f"High dead_ratio ({first_dead.metadata.get('current_dead', 1.0):.2f}) in {first_dead.layer_name}"]
        ))

        return hypotheses
    def _explain_saturated_activations(self) -> List[CausalHypothesis]:
        """Generate hypotheses for saturated activation failures."""
        hypotheses = []

        # Find events with SATURATED state
        sat_events = [e for e in self.events
                     if e.event_type == EventType.ACTIVATION_REGIME_SHIFT
                     and e.to_state == ActivationHealth.SATURATED.value]

        if not sat_events:
            return hypotheses

        first_sat = min(sat_events, key=lambda e: e.step)

        hypotheses.append(CausalHypothesis(
            description=f"Activation saturation detected in layer '{first_sat.layer_name}' at step {first_sat.step}",
            confidence=first_sat.confidence,
            evidence=[first_sat],
            causal_chain=[f"High saturation_ratio ({first_sat.metadata.get('current_saturation', 1.0):.2f}) in {first_sat.layer_name}"]
        ))

        return hypotheses

    def _explain_vanishing_gradients(self) -> List[CausalHypothesis]:
        """Generate hypotheses for vanishing gradient failures."""
        hypotheses = []

        # Find first vanishing gradient event
        vanishing_events = [e for e in self.events
                          if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                          and e.to_state == GradientHealth.VANISHING.value]

        if not vanishing_events:
            return hypotheses

        first_vanishing = min(vanishing_events, key=lambda e: e.step)

        # Hypothesis 1: Originated in this layer
        hypotheses.append(CausalHypothesis(
            description=f"Gradient vanishing originated in layer '{first_vanishing.layer_name}' at step {first_vanishing.step}",
            confidence=first_vanishing.confidence,
            evidence=[first_vanishing],
            causal_chain=[f"Vanishing detected in {first_vanishing.layer_name}"]
        ))

        # Hypothesis 2: Check for activation saturation coupling
        saturation_events = [e for e in self.events
                           if e.event_type == EventType.ACTIVATION_REGIME_SHIFT
                           and e.step <= first_vanishing.step + 10]  # Nearby in time

        if saturation_events:
            nearest_sat = min(saturation_events, key=lambda e: abs(e.step - first_vanishing.step))
            hypotheses.append(CausalHypothesis(
                description=f"Gradient vanishing likely due to LR × activation mismatch - saturation in '{nearest_sat.layer_name}' preceded vanishing",
                confidence=min(first_vanishing.confidence, nearest_sat.confidence),
                evidence=[first_vanishing, nearest_sat],
                causal_chain=[
                    f"Saturation in {nearest_sat.layer_name} at step {nearest_sat.step}",
                    f"Led to vanishing gradients in {first_vanishing.layer_name} at step {first_vanishing.step}"
                ]
            ))

        return hypotheses

    def get_causal_hypotheses(self) -> List[CausalHypothesis]:
        """Get all current causal hypotheses."""
        return self.explain_failure()

    def trace_causal_chain(self, event_type: str) -> List[str]:
        """Trace the causal chain for a specific type of event."""
        # Simple implementation - in practice this would be more sophisticated
        relevant_events = [e for e in self.events if e.event_type.value == event_type]
        return [f"{e.layer_name} at step {e.step}: {e.metadata}" for e in relevant_events]

    def detect_coupled_failures(self, window: int = 5) -> List[Dict[str, Any]]:
        """
        Detect coupled failures (events that occur together or in sequence).
        
        Args:
            window: Maximum step difference to consider events coupled.
            
        Returns:
            List of detected couplings with confidence and direction.
        """
        couplings = []
        if len(self.events) < 2:
            return couplings

        # Sort events by step to find sequential dependencies
        sorted_events = sorted(self.events, key=lambda e: e.step)

        for i, event1 in enumerate(sorted_events):
            for event2 in sorted_events[i+1:]:
                step_diff = event2.step - event1.step
                if step_diff > window:
                    break # Events too far apart

                if event1.layer_name != event2.layer_name:
                    # Potential causal coupling (event1 might influence event2)
                    confidence = min(event1.confidence, event2.confidence)
                    # Boost confidence for specific known patterns (e.g. saturation -> vanishing)
                    if (event1.event_type == EventType.ACTIVATION_REGIME_SHIFT and 
                        event2.event_type == EventType.GRADIENT_HEALTH_TRANSITION):
                        confidence = min(confidence + 0.2, 1.0)

                    couplings.append({
                        'trigger': f"{event1.event_type.value} in {event1.layer_name}",
                        'consequence': f"{event2.event_type.value} in {event2.layer_name}",
                        'step_difference': step_diff,
                        'confidence': confidence,
                        'is_causal_candidate': True
                    })

        return couplings

    def get_root_causes(self) -> List[CausalHypothesis]:
        """Identify and rank root causes based on first-occurrence tracking."""
        hypotheses = []
        for failure_key, layer_name in self.first_failure_layer.items():
            step = self.first_failure_step[failure_key]
            # Find the actual event object
            matching_events = [e for e in self.events if e.layer_name == layer_name and e.step == step]
            evidence = matching_events[:1]
            
            hypotheses.append(CausalHypothesis(
                description=f"Root cause candidate: {failure_key.replace('_', ' ')} originated in '{layer_name}' at step {step}",
                confidence=0.95, # First occurrence is a strong indicator
                evidence=evidence,
                causal_chain=[f"First instance of {failure_key} detected in layer {layer_name}"]
            ))
        return hypotheses

    def _collapse_events(self) -> List[SemanticEvent]:
        """Collapse multiple sequential events in the same layer into a summary trace."""
        # For now, just return all events, but in a production system, 
        # this would merge e.g. HEALTHY->SATURATED and SATURATED->VANISHING
        return self.events

    def export_mermaid_causal_graph(self) -> str:
        """
        Export the captured semantic events as a Mermaid causal graph.
        
        Returns:
            Mermaid-compatible string for visualization
        """
        lines = ["graph TD"]
        
        # Create nodes for all events
        for i, event in enumerate(self.events):
            # Format: EventID["Event Type in Layer (Step X)"]
            label = f"{event.event_type.value} in {event.layer_name} (Step {event.step})"
            lines.append(f'    E{i}["{label}"]')
            
        # Create edges for coupled failures
        couplings = self.detect_coupled_failures()
        for coupling in couplings:
            # Find indices of events (this is simple matching)
            idx1 = -1
            idx2 = -1
            for i, event in enumerate(self.events):
                if f"{event.event_type.value} in {event.layer_name}" == coupling['trigger']:
                    idx1 = i
                if f"{event.event_type.value} in {event.layer_name}" == coupling['consequence']:
                    idx2 = i
            
            if idx1 != -1 and idx2 != -1:
                lines.append(f"    E{idx1} -->|coupled| E{idx2}")
                
        # Create edges for temporal flow in the same layer
        layer_events: Dict[str, List[int]] = {}
        for i, event in enumerate(self.events):
            if event.layer_name not in layer_events:
                layer_events[event.layer_name] = []
            layer_events[event.layer_name].append(i)
            
        for layer, indices in layer_events.items():
            for j in range(len(indices) - 1):
                lines.append(f"    E{indices[j]} -->|temporal| E{indices[j+1]}")
                
        return "\n".join(lines)
