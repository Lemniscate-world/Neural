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


# Try to import the proprietary engine (optional)
try:
    from neuraldbg_engine import CausalEngine

    _HAS_ENGINE = True
except ImportError:
    CausalEngine = None  # type: ignore
    _HAS_ENGINE = False


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

    def __init__(
        self,
        model: nn.Module,
        threshold_vanishing: float = 1e-6,
        threshold_exploding: float = 1e3,
    ):
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
        if hasattr(torch, "_dynamo") and isinstance(
            model, torch._dynamo.eval_frame.OptimizedModule
        ):
            import warnings

            warnings.warn(
                "NeuralDbg: Model is already compiled. Hooks installed after compilation "
                "might not fire in the optimized graph. For best results, wrap the model "
                "with NeuralDbg BEFORE calling torch.compile().",
                UserWarning,
            )

        # Semantic event storage (not tensors!)
        self.events: List[SemanticEvent] = []

        # Causal engine (proprietary, optional)
        self._causal_engine = CausalEngine(self) if _HAS_ENGINE else None

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

    @property
    def _engine(self) -> "CausalEngine":
        if self._causal_engine is None:
            raise NotImplementedError(
                "This feature requires NeuralDBG-Engine (proprietary). "
                "Run: pip install neuraldbg-engine"
            )
        return self._causal_engine

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
        def forward_hook(
            module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
        ):
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
                resource_snapshot, resource_baseline = self._get_step_resource_snapshot(
                    output.device
                )

                # Detect activation regime shifts
                if layer_name in self.previous_activation_stats:
                    prev_stats = self.previous_activation_stats[layer_name]
                    prev_health = self._classify_activation_health(prev_stats)

                    if prev_health != current_health:
                        if current_health != ActivationHealth.NORMAL:
                            self._track_first_occurrence(
                                f"activation_{current_health.value}", layer_name
                            )

                        is_spike, spike_keys = self._is_memory_spike(
                            resource_snapshot, resource_baseline
                        )
                        event = SemanticEvent(
                            event_type=EventType.ACTIVATION_REGIME_SHIFT,
                            layer_name=layer_name,
                            step=self.step,
                            from_state=prev_health.value,
                            to_state=current_health.value,
                            confidence=0.9,
                            metadata={
                                "prev_saturation": prev_stats.get("saturation_ratio"),
                                "current_saturation": activation_stats.get(
                                    "saturation_ratio"
                                ),
                                "prev_dead": prev_stats.get("dead_ratio"),
                                "current_dead": activation_stats.get("dead_ratio"),
                                "resources": resource_snapshot,
                                "memory_spike": is_spike,
                                "memory_spike_keys": spike_keys,
                            },
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
                        metadata={
                            **activation_stats,
                            "resources": resource_snapshot,
                            "memory_spike": False,
                            "memory_spike_keys": [],
                        },
                    )
                    self.events.append(event)
                    if current_health != ActivationHealth.NORMAL:
                        self._track_first_occurrence(
                            f"activation_{current_health.value}", layer_name
                        )

                self.previous_activation_stats[layer_name] = activation_stats

        # Track modules where backward hooks fail so we warn only once
        _backward_hook_failures: Dict[str, bool] = {}

        @dynamo_disable
        def full_backward_hook(
            module: nn.Module,
            grad_input: Tuple[torch.Tensor],
            grad_output: Tuple[torch.Tensor],
        ):
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
                resource_snapshot, resource_baseline = self._get_step_resource_snapshot(
                    grad_tensor.device
                )

                # Detect gradient health transitions
                if layer_name in self.previous_gradient_norms:
                    prev_norm = self.previous_gradient_norms[layer_name]
                    transition = self._detect_gradient_transition(prev_norm, grad_norm)
                    if transition:
                        current_health = self._classify_gradient_health(grad_norm)
                        if current_health != GradientHealth.HEALTHY:
                            self._track_first_occurrence(
                                f"gradient_{current_health.value}", layer_name
                            )

                        is_spike, spike_keys = self._is_memory_spike(
                            resource_snapshot, resource_baseline
                        )
                        event = SemanticEvent(
                            event_type=EventType.GRADIENT_HEALTH_TRANSITION,
                            layer_name=layer_name,
                            step=self.step,
                            from_state=self._classify_gradient_health(prev_norm).value,
                            to_state=current_health.value,
                            confidence=transition["confidence"],
                            metadata={
                                "prev_norm": prev_norm,
                                "current_norm": grad_norm,
                                "transition_type": transition["type"],
                                "resources": resource_snapshot,
                                "memory_spike": is_spike,
                                "memory_spike_keys": spike_keys,
                            },
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
                            "current_norm": grad_norm,
                            "transition_type": "baseline",
                            "resources": resource_snapshot,
                            "memory_spike": False,
                            "memory_spike_keys": [],
                        },
                    )
                    self.events.append(event)
                    if current_health != GradientHealth.HEALTHY:
                        self._track_first_occurrence(
                            f"gradient_{current_health.value}", layer_name
                        )

                self.previous_gradient_norms[layer_name] = grad_norm

        def safe_backward_hook(
            module: nn.Module,
            grad_input: Tuple[torch.Tensor],
            grad_output: Tuple[torch.Tensor],
        ):
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
        if isinstance(
            self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
        ):
            import warnings

            warnings.warn(
                f"NeuralDbg: Model is wrapped in {type(self.model).__name__}. "
                "Hooks might not persist correctly during replication. Consider wrapping "
                "the inner module (.module) instead.",
                UserWarning,
            )

    def _remove_hooks(self):
        """Remove all installed hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _sample_resources(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Snapshot current CPU and (if relevant) GPU memory usage."""
        stats: Dict[str, float] = {}
        if self._psutil_process is not None:
            try:
                stats["cpu_memory_mb"] = (
                    self._psutil_process.memory_info().rss / 1024**2
                )
            except Exception:
                self._psutil_process = None
        if device is not None and device.type == "cuda":
            stats["gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated(device) / 1024**2
            )
            stats["gpu_memory_reserved_mb"] = (
                torch.cuda.memory_reserved(device) / 1024**2
            )
        return stats

    def _get_step_resource_snapshot(
        self, device: Optional[torch.device] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return (current_snapshot, baseline) for this step, sampling at most once per step."""
        if (
            self._resource_snapshot_cache is not None
            and self._resource_snapshot_cache[0] == self.step
        ):
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
            "mean": t_float.mean().item(),
            "std": t_float.std().item(),
            "min": t_float.min().item(),
            "max": t_float.max().item(),
            "sparsity": sparsity,
            "dead_ratio": dead_ratio,
            "norm": t_float.norm().item(),
            "saturation_ratio": saturation_ratio,
        }

    def _classify_activation_health(self, stats: Dict[str, float]) -> ActivationHealth:
        """Classify activation regime — delegates to engine when available."""
        return self._engine.activation.classify_health(stats)

    def _detect_activation_shift(
        self, prev_stats: Dict[str, float], current_stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Detect activation shifts — delegates to engine."""
        return self._engine.activation.detect_shift(prev_stats, current_stats)

    def _classify_gradient_health(self, norm: float) -> GradientHealth:
        """Classify gradient health — delegates to engine."""
        return self._engine.gradient.classify_health(norm)

    def _track_first_occurrence(self, failure_type: str, layer_name: str):
        """Track the first layer that encountered a specific failure."""
        if failure_type not in self.first_failure_step:
            self.first_failure_step[failure_type] = self.step
            self.first_failure_layer[failure_type] = layer_name

    def _detect_gradient_transition(
        self, prev_norm: float, current_norm: float
    ) -> Optional[Dict[str, Any]]:
        """Detect transitions in gradient health — delegates to engine."""
        return self._engine.gradient.detect_transition(prev_norm, current_norm)

    def explain_failure(
        self, failure_type: str = "vanishing_gradients"
    ) -> List[CausalHypothesis]:
        """Provide ranked causal hypotheses — delegates to engine."""
        return self._engine.explain.explain(failure_type)

    def _explain_vanishing_gradients(self) -> List[CausalHypothesis]:
        """Generate hypotheses for vanishing — delegates to engine."""
        return self._engine.explain._explain_vanishing_gradients()

    def _explain_exploding_gradients(self) -> List[CausalHypothesis]:
        """Generate hypotheses for exploding — delegates to engine."""
        return self._engine.explain._explain_exploding_gradients()

    def _explain_dead_neurons(self) -> List[CausalHypothesis]:
        """Generate hypotheses for dead neurons — delegates to engine."""
        return self._engine.explain._explain_dead_neurons()

    def _explain_saturated_activations(self) -> List[CausalHypothesis]:
        """Generate hypotheses for saturated activations — delegates to engine."""
        return self._engine.explain._explain_saturated_activations()

    def get_causal_hypotheses(self) -> List[CausalHypothesis]:
        """Get all current causal hypotheses — delegates to engine."""
        return self._engine.explain.get_causal_hypotheses()

    def trace_causal_chain(self, event_type: str) -> List[str]:
        """Trace the causal chain — delegates to engine."""
        return self._engine.explain.trace_causal_chain(event_type)

    def detect_coupled_failures(self, window: int = 5) -> List[Dict[str, Any]]:
        """Detect coupled failures — delegates to engine."""
        return self._engine.coupling.detect(window)

    def get_root_causes(self) -> List[CausalHypothesis]:
        """Identify root causes — delegates to engine."""
        return self._engine.explain.get_root_causes()

    def _event_matches_failure_key(
        self,
        event: SemanticEvent,
        failure_key: str,
    ) -> bool:
        """Check event/failure key match — delegates to engine."""
        return self._engine.explain.event_matches_failure_key(event, failure_key)

    def _classify_data_health(
        self, tensor: torch.Tensor
    ) -> Tuple[DataHealth, Dict[str, Any]]:
        """Classify data health — delegates to engine."""
        return self._engine.data.classify_health(tensor)

    def _check_data_anomaly(self, tensor: torch.Tensor, layer_name: str):
        """Detect data anomalies — delegates to engine."""
        return self._engine.data.check_anomaly(tensor, layer_name)

    def _explain_optimizer_instability(self) -> List[CausalHypothesis]:
        """Generate hypotheses for optimizer instability — delegates to engine."""
        return self._engine.explain._explain_optimizer_instability()

    def _explain_data_anomaly(self) -> List[CausalHypothesis]:
        """Generate hypotheses for data anomaly — delegates to engine."""
        return self._engine.explain._explain_data_anomaly()

    def _collapse_events(self) -> List[SemanticEvent]:
        """Collapse sequential events — delegates to engine."""
        return self._engine.explain.collapse_events()

    def export_aquarium_package(self, package_path: str) -> str:
        """Export JSON package for Aquarium — delegates to engine."""
        return self._engine.explain.export_aquarium_package(package_path)

    def export_mermaid_causal_graph(self) -> str:
        """Export Mermaid causal graph — delegates to engine."""
        return self._engine.explain.export_mermaid_causal_graph()
