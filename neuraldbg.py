"""
NeuralDbg Causal Inference Engine

This module defines the NeuralDbg class, which is a causal inference engine for deep learning training dynamics.
It extracts semantic events from training, compresses them into causal patterns, and provides
post-mortem reasoning about training failures.
"""

import math
import json
from pathlib import Path

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
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

class OptimizerHealth(Enum):
    """Optimizer stability states."""
    STABLE = "stable"
    LOSS_PLATEAU = "loss_plateau"
    LOSS_SPIKE = "loss_spike"
    DIVERGING = "diverging"

class DataHealth(Enum):
    """Data quality states for anomaly detection."""
    NORMAL = "normal"
    NAN_DETECTED = "nan_detected"
    INF_DETECTED = "inf_detected"
    DISTRIBUTION_SHIFT = "distribution_shift"

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

        # Optimizer instability tracking
        self.loss_history: List[float] = []
        self.previous_optimizer_health: OptimizerHealth = OptimizerHealth.STABLE

        # Data anomaly tracking
        self.previous_input_stats: Dict[str, Dict[str, float]] = {}
        self.previous_data_health: Dict[str, DataHealth] = {}

        # Pre-computed module-to-name mapping for O(1) lookup
        self._module_names: Dict[int, str] = {}
        for name, mod in self.model.named_modules():
            self._module_names[id(mod)] = name or "root"

        # Causal tracking: First layer to fail in a specific way
        self.first_failure_step: Dict[str, int] = {}  # failure_key -> step
        self.first_failure_layer: Dict[str, str] = {}  # failure_key -> layer_name

        # Hook storage for automatic monitoring
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Training state
        self.step = 0
        self.is_monitoring = False

        # Resource profiling — psutil is optional but treated as a runtime dep
        self._psutil_process = None
        try:
            import psutil as _psutil
            self._psutil_process = _psutil.Process()
        except ImportError:
            pass

        # Per-step resource snapshot cache: (step, snapshot) — sampled once per step
        self._resource_snapshot_cache: Optional[Tuple[int, Dict[str, float]]] = None
        # Snapshot from the previous step used as spike baseline
        self._resource_baseline: Dict[str, float] = {}

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

    def record_loss(self, loss_value: float):
        """Record a loss value for optimizer instability detection.

        Call this after each training step so the engine can track
        loss plateaus, spikes, and divergence.

        Args:
            loss_value: The scalar loss value for the current step.
        """
        self.loss_history.append(loss_value)
        current_health = self._classify_optimizer_health()

        if current_health != self.previous_optimizer_health:
            if current_health != OptimizerHealth.STABLE:
                self._track_first_occurrence(
                    f"optimizer_{current_health.value}", "optimizer"
                )

            event = SemanticEvent(
                event_type=EventType.OPTIMIZER_INSTABILITY,
                layer_name="optimizer",
                step=self.step,
                from_state=self.previous_optimizer_health.value,
                to_state=current_health.value,
                confidence=0.85,
                metadata={
                    "recent_losses": self.loss_history[-10:],
                    "current_loss": loss_value,
                },
            )
            self.events.append(event)
            self.previous_optimizer_health = current_health

    def _classify_optimizer_health(self) -> OptimizerHealth:
        """Classify optimizer stability from loss history."""
        history = self.loss_history
        if len(history) < 3:
            return OptimizerHealth.STABLE

        recent = history[-5:] if len(history) >= 5 else history

        # Check for NaN/Inf (divergence)
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return OptimizerHealth.DIVERGING

        # Check for loss spike: latest loss > 10x the mean of previous values
        if len(history) >= 5:
            prev_mean = sum(history[-6:-1]) / len(history[-6:-1])
            if prev_mean > 0 and recent[-1] > prev_mean * 10:
                return OptimizerHealth.LOSS_SPIKE

        # Check for plateau: std of recent losses is near zero relative to mean
        if len(recent) >= 3:
            mean_val = sum(recent) / len(recent)
            variance = sum((v - mean_val) ** 2 for v in recent) / len(recent)
            std_val = math.sqrt(variance)
            if mean_val != 0 and std_val / abs(mean_val) < 1e-4:
                return OptimizerHealth.LOSS_PLATEAU

        return OptimizerHealth.STABLE

    def _install_hooks(self):
        """Install forward and backward hooks to extract semantic events."""
        @dynamo_disable
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
            """Extract semantic events from forward pass."""
            if not self.is_monitoring:
                return

            layer_name = self._get_layer_name(module)

            # --- Data anomaly detection on inputs ---
            if input and len(input) > 0 and isinstance(input[0], torch.Tensor):
                self._check_data_anomaly(input[0], layer_name)

            # Extract activation regime information
            if isinstance(output, torch.Tensor):
                activation_stats = self._compute_activation_stats(output)
                current_health = self._classify_activation_health(activation_stats)

                # Sample resources once per step (outside transition check to build baseline)
                resource_snapshot, resource_baseline = self._get_step_resource_snapshot(output.device)

                # Detect activation regime shifts
                if layer_name in self.previous_activation_stats:
                    prev_stats = self.previous_activation_stats[layer_name]
                    prev_health = self._classify_activation_health(prev_stats)

                    if prev_health != current_health:
                        if current_health != ActivationHealth.NORMAL:
                            self._track_first_occurrence(f"activation_{current_health.value}", layer_name)

                        is_spike, spike_keys = self._is_memory_spike(resource_snapshot, resource_baseline)
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
                                'current_dead': activation_stats.get('dead_ratio'),
                                'resources': resource_snapshot,
                                'memory_spike': is_spike,
                                'memory_spike_keys': spike_keys,
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

        # Track modules where backward hooks fail so we warn only once
        _backward_hook_failures: Dict[str, bool] = {}

        @dynamo_disable
        def full_backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            """Extract semantic events from backward pass using full_backward_hook."""
            if not self.is_monitoring:
                return

            layer_name = self._get_layer_name(module)

            # Extract gradient health information
            # In full_backward_hook, grad_output is a tuple of gradients w.r.t. outputs
            if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                grad_tensor = grad_output[0]
                grad_norm = grad_tensor.norm().item()

                # Sample resources once per step (outside transition check to build baseline)
                resource_snapshot, resource_baseline = self._get_step_resource_snapshot(grad_tensor.device)

                # Detect gradient health transitions
                if layer_name in self.previous_gradient_norms:
                    prev_norm = self.previous_gradient_norms[layer_name]
                    transition = self._detect_gradient_transition(prev_norm, grad_norm)
                    if transition:
                        current_health = self._classify_gradient_health(grad_norm)
                        if current_health != GradientHealth.HEALTHY:
                            self._track_first_occurrence(f"gradient_{current_health.value}", layer_name)

                        is_spike, spike_keys = self._is_memory_spike(resource_snapshot, resource_baseline)
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
                                'transition_type': transition['type'],
                                'resources': resource_snapshot,
                                'memory_spike': is_spike,
                                'memory_spike_keys': spike_keys,
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

        def safe_backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            """Wrapper that catches inplace-operation errors from full_backward_hook.

            Models with inplace operations (e.g., ReLU(inplace=True) in
            ResNet/BatchNorm) raise RuntimeError when full_backward_hook
            interacts with views modified inplace. This wrapper degrades
            gracefully: forward hooks still capture activation and data
            anomaly events, only gradient tracking is lost for affected
            modules.
            """
            try:
                return full_backward_hook(module, grad_input, grad_output)
            except RuntimeError:
                layer_name = self._get_layer_name(module)
                if layer_name not in _backward_hook_failures:
                    _backward_hook_failures[layer_name] = True
                    import warnings
                    warnings.warn(
                        f"NeuralDbg: Backward hook failed for '{layer_name}' "
                        f"(likely inplace operation). Gradient tracking disabled "
                        f"for this module. Forward hooks still active.",
                        UserWarning,
                    )
                return None

        # Install hooks on leaf modules for maximum compatibility with
        # torch.compile. We use register_backward_hook (not
        # register_full_backward_hook) because the "full" variant wraps
        # module outputs in a BackwardHookFunction view. Any downstream
        # inplace operation on that view (e.g., `out += identity` in ResNet
        # residual connections, or ReLU(inplace=True)) triggers a
        # RuntimeError from PyTorch's autograd engine. The older
        # register_backward_hook does NOT wrap outputs, so it is compatible
        # with all model architectures including ResNet, EfficientNet, etc.
        # Trade-off: register_backward_hook may not fire for modules whose
        # inputs don't require grad, but those modules are irrelevant for
        # gradient health monitoring anyway.
        for name, module in self.model.named_modules():
            # Skip non-leaf modules (except root) to avoid redundant captures
            if len(list(module.children())) > 0 and name != "":
                continue
                
            self.hooks.append(module.register_forward_hook(forward_hook))
            self.hooks.append(module.register_backward_hook(safe_backward_hook))

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

    def _sample_resources(self, device: Optional[torch.device] = None) -> Dict[str, float]:
        """Snapshot current CPU and (if relevant) GPU memory usage."""
        stats: Dict[str, float] = {}
        if self._psutil_process is not None:
            try:
                stats['cpu_memory_mb'] = self._psutil_process.memory_info().rss / 1024 ** 2
            except Exception:
                pass
        if device is not None and device.type == 'cuda':
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(device) / 1024 ** 2
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved(device) / 1024 ** 2
        return stats

    def _get_step_resource_snapshot(self, device: Optional[torch.device] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (current_snapshot, baseline) for this step, sampling at most once per step."""
        if self._resource_snapshot_cache is not None and self._resource_snapshot_cache[0] == self.step:
            return self._resource_snapshot_cache[1], self._resource_baseline
        # New step: promote previous snapshot to baseline, then take a fresh one
        if self._resource_snapshot_cache is not None:
            self._resource_baseline = self._resource_snapshot_cache[1]
        snapshot = self._sample_resources(device)
        self._resource_snapshot_cache = (self.step, snapshot)
        return snapshot, self._resource_baseline

    def _is_memory_spike(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """Detect spikes: >20 % relative rise AND >50 MB absolute vs previous-step baseline."""
        spike_keys: List[str] = []
        for key, curr_val in current.items():
            if key not in baseline:
                continue
            prev_val = baseline[key]
            delta = curr_val - prev_val
            if prev_val > 0 and (delta / prev_val) > 0.20 and delta > 50.0:
                spike_keys.append(key)
        return bool(spike_keys), spike_keys

    def _get_layer_name(self, module: nn.Module) -> str:
        """Get the name of a module from the pre-computed mapping (O(1) lookup)."""
        name = self._module_names.get(id(module))
        if name is not None:
            return name

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
        elif failure_type == "optimizer_instability":
            hypotheses.extend(self._explain_optimizer_instability())
        elif failure_type == "data_anomaly":
            hypotheses.extend(self._explain_data_anomaly())

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
            causal_chain=[
                "High saturation_ratio "
                f"({first_sat.metadata.get('current_saturation', first_sat.metadata.get('saturation_ratio', 1.0)):.2f}) "
                f"in {first_sat.layer_name}"
            ]
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
                           and e.to_state == ActivationHealth.SATURATED.value
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
            matching_events = [
                e for e in self.events
                if e.layer_name == layer_name
                and e.step == step
                and self._event_matches_failure_key(e, failure_key)
            ]
            if not matching_events:
                matching_events = [
                    e for e in self.events
                    if e.layer_name == layer_name and e.step == step
                ]
            evidence = matching_events[:1]
            
            hypotheses.append(CausalHypothesis(
                description=f"Root cause candidate: {failure_key.replace('_', ' ')} originated in '{layer_name}' at step {step}",
                confidence=0.95, # First occurrence is a strong indicator
                evidence=evidence,
                causal_chain=[f"First instance of {failure_key} detected in layer {layer_name}"]
            ))
        return hypotheses

    def _event_matches_failure_key(
        self,
        event: SemanticEvent,
        failure_key: str,
    ) -> bool:
        """Return whether an event is the evidence for a tracked failure key."""
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
                event.event_type == EventType.DATA_ANOMALY
                and event.to_state == state
            )
        return False

    def _classify_data_health(self, tensor: torch.Tensor) -> Tuple[DataHealth, Dict[str, Any]]:
        """Classify data health and return state + metadata.

        Returns a tuple of (health_state, metadata_dict) based on the tensor contents.
        NaN takes priority over Inf, which takes priority over distribution shift.
        """
        t = tensor.detach().float()

        has_nan = torch.isnan(t).any().item()
        if has_nan:
            return DataHealth.NAN_DETECTED, {"nan_count": int(torch.isnan(t).sum().item())}

        has_inf = torch.isinf(t).any().item()
        if has_inf:
            return DataHealth.INF_DETECTED, {"inf_count": int(torch.isinf(t).sum().item())}

        return DataHealth.NORMAL, {}

    def _check_data_anomaly(self, tensor: torch.Tensor, layer_name: str):
        """Detect data anomalies (NaN, Inf, distribution shifts) in input tensors.

        Uses transition-based tracking like gradient and activation health:
        only emits events when the data health state changes for a given layer.
        """
        current_health, health_metadata = self._classify_data_health(tensor)

        # Distribution shift detection (only when current health is NORMAL)
        # Skip mean/std computation for NaN/Inf tensors to avoid poisoning stats
        if current_health not in (DataHealth.NAN_DETECTED, DataHealth.INF_DETECTED):
            t = tensor.detach().float()
            current_mean = t.mean().item()
            current_std = t.std().item()

            if layer_name in self.previous_input_stats:
                prev = self.previous_input_stats[layer_name]
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

            # Only update stats with clean values (not NaN/Inf)
            self.previous_input_stats[layer_name] = {
                "mean": current_mean,
                "std": current_std,
            }

        # Transition detection: only emit event when health state changes
        prev_health = self.previous_data_health.get(layer_name, DataHealth.NORMAL)

        if current_health != prev_health:
            if current_health != DataHealth.NORMAL:
                self._track_first_occurrence(f"data_{current_health.value}", layer_name)

            confidence = 1.0
            if current_health == DataHealth.DISTRIBUTION_SHIFT:
                mean_shift_val = health_metadata.get("mean_shift_sigma", 3.0)
                confidence = min(mean_shift_val * 0.2, 1.0)

            self.events.append(SemanticEvent(
                event_type=EventType.DATA_ANOMALY,
                layer_name=layer_name,
                step=self.step,
                from_state=prev_health.value,
                to_state=current_health.value,
                confidence=confidence,
                metadata=health_metadata,
            ))

        self.previous_data_health[layer_name] = current_health

    def _explain_optimizer_instability(self) -> List[CausalHypothesis]:
        """Generate hypotheses for optimizer instability failures."""
        hypotheses: List[CausalHypothesis] = []

        instability_events = [
            e for e in self.events
            if e.event_type == EventType.OPTIMIZER_INSTABILITY
        ]

        if not instability_events:
            return hypotheses

        first_event = min(instability_events, key=lambda e: e.step)

        description_map = {
            OptimizerHealth.LOSS_SPIKE.value: (
                f"Loss spike detected at step {first_event.step}. "
                "Possible causes: learning rate too high, corrupted batch, or "
                "gradient explosion propagating to the loss."
            ),
            OptimizerHealth.LOSS_PLATEAU.value: (
                f"Loss plateau detected at step {first_event.step}. "
                "Training may be stuck in a local minimum or the learning rate "
                "is too small to make progress."
            ),
            OptimizerHealth.DIVERGING.value: (
                f"Training divergence (NaN/Inf loss) detected at step "
                f"{first_event.step}. Check for numerical instability, "
                "learning rate explosion, or corrupted data."
            ),
        }

        desc = description_map.get(
            first_event.to_state,
            f"Optimizer instability detected at step {first_event.step}",
        )

        hypotheses.append(CausalHypothesis(
            description=desc,
            confidence=first_event.confidence,
            evidence=[first_event],
            causal_chain=[
                f"Optimizer transitioned from {first_event.from_state} to "
                f"{first_event.to_state} at step {first_event.step}"
            ],
        ))

        # Cross-reference: if gradient explosion preceded the spike
        if first_event.to_state in (
            OptimizerHealth.LOSS_SPIKE.value,
            OptimizerHealth.DIVERGING.value,
        ):
            exploding_before = [
                e for e in self.events
                if e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                and e.to_state == GradientHealth.EXPLODING.value
                and e.step <= first_event.step
            ]
            if exploding_before:
                grad_event = max(exploding_before, key=lambda e: e.step)
                hypotheses.append(CausalHypothesis(
                    description=(
                        f"Loss {first_event.to_state} at step "
                        f"{first_event.step} was likely caused by gradient "
                        f"explosion in '{grad_event.layer_name}' at step "
                        f"{grad_event.step}."
                    ),
                    confidence=min(
                        first_event.confidence + 0.1, 1.0
                    ),
                    evidence=[first_event, grad_event],
                    causal_chain=[
                        f"Gradient explosion in {grad_event.layer_name} "
                        f"at step {grad_event.step}",
                        f"Led to {first_event.to_state} at step "
                        f"{first_event.step}",
                    ],
                ))

        return hypotheses

    def _explain_data_anomaly(self) -> List[CausalHypothesis]:
        """Generate hypotheses for data anomaly failures."""
        hypotheses: List[CausalHypothesis] = []

        anomaly_events = [
            e for e in self.events
            if e.event_type == EventType.DATA_ANOMALY
        ]

        if not anomaly_events:
            return hypotheses

        first_event = min(anomaly_events, key=lambda e: e.step)

        description_map = {
            DataHealth.NAN_DETECTED.value: (
                f"NaN values detected in input to layer "
                f"'{first_event.layer_name}' at step {first_event.step}. "
                "This indicates corrupted data or upstream numerical overflow."
            ),
            DataHealth.INF_DETECTED.value: (
                f"Inf values detected in input to layer "
                f"'{first_event.layer_name}' at step {first_event.step}. "
                "Check for division by zero or unbounded feature values."
            ),
            DataHealth.DISTRIBUTION_SHIFT.value: (
                f"Input distribution shift detected at layer "
                f"'{first_event.layer_name}' at step {first_event.step}. "
                "The input statistics changed significantly, which may "
                "destabilize training."
            ),
        }

        desc = description_map.get(
            first_event.to_state,
            f"Data anomaly detected at layer '{first_event.layer_name}' "
            f"at step {first_event.step}",
        )

        hypotheses.append(CausalHypothesis(
            description=desc,
            confidence=first_event.confidence,
            evidence=[first_event],
            causal_chain=[
                f"Data anomaly ({first_event.to_state}) in "
                f"{first_event.layer_name} at step {first_event.step}"
            ],
        ))

        return hypotheses

    def _collapse_events(self) -> List[SemanticEvent]:
        """Collapse sequential events in the same layer into summary traces.

        Merges chains like HEALTHY->SATURATED followed by SATURATED->VANISHING
        in the same layer into a single HEALTHY->VANISHING summary event,
        preserving the step range and combining metadata.

        Baseline events (from_state="NONE") are always kept individually since
        they represent initial state capture, not transitions.
        """
        if not self.events:
            return []

        # Separate baseline events from transition events
        baseline_events: List[SemanticEvent] = []
        transition_events: List[SemanticEvent] = []
        for event in self.events:
            if event.from_state == "NONE":
                baseline_events.append(event)
            else:
                transition_events.append(event)

        # Group transition events by (layer_name, event_type)
        groups: Dict[Tuple[str, str], List[SemanticEvent]] = {}
        for event in transition_events:
            key = (event.layer_name, event.event_type.value)
            if key not in groups:
                groups[key] = []
            groups[key].append(event)

        collapsed: List[SemanticEvent] = list(baseline_events)
        for key, group in groups.items():
            sorted_group = sorted(group, key=lambda e: e.step)

            if len(sorted_group) <= 1:
                collapsed.extend(sorted_group)
                continue

            # Check for ANY reversion within the chain (not just first vs last)
            # A reversion means a state appeared as to_state and later as from_state
            has_reversion = False
            seen_to_states = set()
            for event in sorted_group:
                if event.to_state in seen_to_states:
                    # A state we transitioned away from has reappeared
                    has_reversion = True
                    break
                seen_to_states.add(event.from_state)

            if has_reversion:
                # States reverted at some point, keep all individual events
                collapsed.extend(sorted_group)
            else:
                first = sorted_group[0]
                last = sorted_group[-1]
                merged_metadata = dict(first.metadata)
                merged_metadata["collapsed_count"] = len(sorted_group)
                merged_metadata["step_range"] = (
                    f"{first.step}-{last.step}"
                )

                collapsed.append(SemanticEvent(
                    event_type=first.event_type,
                    layer_name=first.layer_name,
                    step=first.step,
                    from_state=first.from_state,
                    to_state=last.to_state,
                    confidence=max(e.confidence for e in sorted_group),
                    metadata=merged_metadata,
                ))

        # Sort by step for consistent ordering
        collapsed.sort(key=lambda e: e.step)
        return collapsed

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert internal values into JSON-serializable primitives."""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): NeuralDbg._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [NeuralDbg._json_safe(v) for v in value]
        return value

    @classmethod
    def _event_to_dict(cls, event: SemanticEvent) -> Dict[str, Any]:
        """Serialize a semantic event for external tools."""
        return {
            "event_type": cls._json_safe(event.event_type),
            "layer_name": event.layer_name,
            "step": event.step,
            "from_state": str(cls._json_safe(event.from_state)),
            "to_state": str(cls._json_safe(event.to_state)),
            "confidence": event.confidence,
            "metadata": cls._json_safe(event.metadata),
        }

    @classmethod
    def _hypothesis_to_dict(cls, hypothesis: CausalHypothesis) -> Dict[str, Any]:
        """Serialize a causal hypothesis for external tools."""
        return {
            "description": hypothesis.description,
            "confidence": hypothesis.confidence,
            "evidence": [cls._event_to_dict(event) for event in hypothesis.evidence],
            "causal_chain": cls._json_safe(hypothesis.causal_chain),
        }

    def export_aquarium_package(self, package_path: str) -> str:
        """Export a compact JSON package for Aquarium-style IDE consumers."""
        output_dir = Path(package_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        hypotheses = self.get_causal_hypotheses()
        root_causes = self.get_root_causes()
        package = {
            "step": self.step,
            "events": [self._event_to_dict(event) for event in self.events],
            "hypotheses": [
                self._hypothesis_to_dict(hypothesis)
                for hypothesis in hypotheses
            ],
            "root_causes": [
                self._hypothesis_to_dict(hypothesis)
                for hypothesis in root_causes
            ],
            "couplings": self._json_safe(self.detect_coupled_failures()),
        }

        output_file = output_dir / "events.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(package, f, indent=2)
        return str(output_file)

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
