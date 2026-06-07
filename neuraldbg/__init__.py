"""
NeuralDbg Causal Inference Engine

This module defines the NeuralDbg class, a causal inference engine for deep learning
training dynamics.
It extracts semantic events from training, compresses them into causal patterns, and provides
post-mortem reasoning about training failures.
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

__version__ = "1.3.2"

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
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class CausalHypothesis:
    """A ranked hypothesis about the cause of a training failure."""

    description: str
    confidence: float
    evidence: List[SemanticEvent]
    causal_chain: List[str]


class TensorDiskCache:
    """Manages caching of large tensors on disk to prevent RAM/VRAM OOMs."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or "artifacts/tensor_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._files: List[Path] = []

    def save(self, tensor: torch.Tensor, prefix: str = "tensor") -> str:
        """Save a tensor to disk and return the absolute file path."""
        filename = f"{prefix}_{uuid.uuid4().hex}.pt"
        filepath = self.cache_dir / filename
        # Detach and move to CPU to release GPU VRAM immediately
        torch.save(tensor.detach().cpu(), filepath)
        self._files.append(filepath)
        return str(filepath.absolute().as_posix())

    def cleanup(self):
        """Delete all cached files and clear the registry."""
        for filepath in self._files:
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception:
                    pass
        self._files.clear()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


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
        # Format: "ClassName_index" (e.g. "Linear_0", "Tanh_1") for readability
        self._module_names: Dict[int, str] = {}
        for name, mod in self.model.named_modules():
            readable = f"{type(mod).__name__}_{name}" if name else "root"
            self._module_names[id(mod)] = readable

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

        # Disk cache for storing large tensors
        self.disk_cache = TensorDiskCache()

        # Per-step resource snapshot cache: (step, snapshot) — sampled once per step
        self._resource_snapshot_cache: Optional[Tuple[int, Dict[str, float]]] = None
        # Snapshot from the previous step used as baseline
        self._resource_baseline: Dict[str, float] = {}

        # FIX-001 — Composite-module support (BUG-001 / pytorch#41508):
        # users can register hooks on non-leaf modules (e.g. nn.MultiheadAttention)
        # via the new register_composite_hook() public API. This fills the gap
        # revealed by pytorch/pytorch#41508 where MHA gradients go undetected
        # because MHA is a composite module with no leaf children.
        self._composite_modules: List[nn.Module] = []

        # FIX-001 — Silent-loss detection: track steps with NO gradient
        # event at all. If the user calls loss.backward() but NeuralDBG
        # never sees a gradient_health_transition event, the model is
        # probably a fully-composite architecture (e.g. MHA, custom fused
        # kernels) and the auto-leaf hooks are silent. Flag it on __exit__.
        self._steps_without_gradient_events: int = 0
        self._silent_loss_warning_emitted: bool = False

        # FIX-001 — Track how many leaf modules actually got hooks. If
        # zero, the model is fully composite and we warn at __enter__.
        self._hooked_leaf_count: int = 0

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

        # FIX-001 — Warn the user when the wrapped model exposes no
        # *internal* leaf modules (i.e., all leaves the auto installer
        # could attach to are the model root itself). This usually means
        # the architecture is fully composite (e.g. a bare
        # nn.MultiheadAttention, a custom fused block, a custom
        # autograd Function) and the auto hooks will only see the root
        # forward/backward. Internal parameters (e.g. MHA's in_proj_*)
        # are therefore blind to NeuralDBG unless the user opts in via
        # register_composite_hook().
        if self._hooked_leaf_count <= 1 and not self._composite_modules:
            # Heuristic: at most 1 leaf means the only "leaf" the walker
            # found is the root itself. Internal params exist on the root
            # and bypass any hook on the root.
            import warnings

            warnings.warn(
                "NeuralDbg: this model exposes no internal leaf modules. "
                "Auto-installed forward/backward hooks will NOT see internal "
                "parameters (e.g. nn.MultiheadAttention's in_proj_weight, "
                "custom fused kernels, custom autograd Functions). To "
                "instrument composite modules manually, call "
                "`dbg.register_composite_hook(module)` after `with NeuralDbg(...)`. "
                "See docs/blog/2026-06-13-pytorch-41508-postmortem.html for "
                "a worked example.",
                UserWarning,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and cleanup."""
        # FIX-001 — Silent-loss detection: if the user has called
        # loss.backward() at least 3 times but NeuralDBG never saw a
        # gradient_health_transition event on any monitored module, the
        # model is probably fully composite and our hooks are blind.
        if (
            not self._silent_loss_warning_emitted
            and self.step >= 3
            and not any(
                e.event_type == EventType.GRADIENT_HEALTH_TRANSITION
                for e in self.events
            )
        ):
            import warnings

            warnings.warn(
                f"NeuralDbg: {self.step} training step(s) executed but no "
                "gradient_health_transition event was captured. This usually "
                "means the model is fully composite (e.g. nn.MultiheadAttention, "
                "custom fused kernels) and the auto leaf-module hooks are silent. "
                "Use `dbg.register_composite_hook(attn_module)` to instrument "
                "composite modules explicitly. "
                "See docs/blog/2026-06-13-pytorch-41508-postmortem.html.",
                UserWarning,
            )
            self._silent_loss_warning_emitted = True

        self._remove_hooks()
        self.disk_cache.cleanup()
        self.is_monitoring = False

    def register_composite_hook(self, module: nn.Module) -> None:
        """Manually install NeuralDbg hooks on a composite (non-leaf) module.

        Use this to instrument modules that the auto leaf-only hook
        installer would skip, such as ``nn.MultiheadAttention`` or any
        custom fused/quantised block that has no leaf children.

        The module is added to the internal ``_composite_modules`` list
        and the same forward/backward hook pair used by the auto
        installer is attached. Hooks are removed automatically on
        ``__exit__`` via the standard ``_remove_hooks`` path.

        Args:
            module: The composite ``nn.Module`` to instrument.

        Example:
            >>> with NeuralDbg(model) as dbg:
            ...     dbg.register_composite_hook(attn)  # nn.MultiheadAttention
            ...     loss = train_step(x, y)
            ...     loss.backward()
            ...     dbg.record_loss(loss.item())

        Added in FIX-001 (v1.3.2) — see BUG-001 / POST-001
        (post-mortem of pytorch/pytorch#41508).
        """
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"register_composite_hook expects nn.Module, got {type(module).__name__}"
            )
        # The auto installer skips non-leaf modules. We bypass that here.
        # We re-use the exact same hook pair by calling _install_hooks()
        # logic on this single module.
        for name, sub in self.model.named_modules():
            if id(sub) == id(module):
                self._install_hook_on_module(sub, name)
                self._composite_modules.append(module)
                return
        # Module not found in the model's tree; warn but still install
        # the hooks so the user gets a chance to debug.
        import warnings

        warnings.warn(
            f"NeuralDbg: register_composite_hook target '{type(module).__name__}' "
            "was not found inside the wrapped model. Hooks are still installed "
            "but events will be tagged with the module's class name only.",
            UserWarning,
        )
        self._install_hook_on_module(module, type(module).__name__)
        self._composite_modules.append(module)

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
        """Install forward and backward hooks to extract semantic events.

        Defines the hook closures, stashes them on ``self`` so that
        :meth:`register_composite_hook` and :meth:`_install_hook_on_module`
        can re-use the exact same hook pair on non-leaf modules, and then
        walks the model to attach the hooks on every leaf module.
        """

        @dynamo_disable
        def forward_hook(
            module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
        ):
            """Extract semantic events from forward pass."""
            if not self.is_monitoring:
                return

            layer_name = self._get_layer_name(module)

            # --- Data anomaly detection on inputs AND outputs ---
            if input and len(input) > 0 and isinstance(input[0], torch.Tensor):
                try:
                    self._check_data_anomaly(input[0], layer_name)
                except NotImplementedError:
                    pass
            if isinstance(output, torch.Tensor):
                try:
                    self._check_data_anomaly(output, layer_name)
                except NotImplementedError:
                    pass

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

        # FIX-001 — Stash the hook implementations on self so
        # register_composite_hook() and _install_hook_on_module() can
        # re-use the exact same pair on non-leaf (composite) modules.
        self._forward_hook_impl = forward_hook
        self._backward_hook_impl = safe_backward_hook

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
            self._install_hook_on_module(module, name)
            self._hooked_leaf_count += 1

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

    def _install_hook_on_module(self, module: "nn.Module", name: str) -> None:
        """Attach the forward/backward hook pair to a single module.

        Used by :meth:`_install_hooks` for the leaf-module walk and by
        :meth:`register_composite_hook` to opt-in instrument composite
        modules that the auto installer skipped (FIX-001 / BUG-001).

        The hook implementations must already be defined on
        ``self._forward_hook_impl`` and ``self._backward_hook_impl``;
        this is done at the start of :meth:`_install_hooks`.
        """
        forward_hook = getattr(self, "_forward_hook_impl", None)
        backward_hook = getattr(self, "_backward_hook_impl", None)
        if forward_hook is None or backward_hook is None:
            # Hooks not yet defined (e.g. _install_hooks was never called).
            # Fall back to running _install_hooks first so the closures
            # exist; then attach on the target module.
            self._install_hooks()
            forward_hook = self._forward_hook_impl
            backward_hook = self._backward_hook_impl
        self.hooks.append(module.register_forward_hook(forward_hook))
        self.hooks.append(module.register_backward_hook(backward_hook))

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
        t = tensor.detach()
        if not torch.is_floating_point(t):
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sparsity": 0.0,
                "dead_ratio": 0.0,
                "norm": 0.0,
                "saturation_ratio": 0.0,
            }

        numel = t.numel()
        if numel == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sparsity": 0.0,
                "dead_ratio": 0.0,
                "norm": 0.0,
                "saturation_ratio": 0.0,
            }

        # Determine epsilon based on dtype to avoid underflow
        eps = 1e-4 if t.dtype in (torch.float16, torch.bfloat16) else 1e-9

        # Calculate sparsity (fraction of zeros)
        # Using a small epsilon for float comparison, sum booleans to avoid float tensor allocations
        sparsity = (t.abs() < eps).sum().item() / numel

        # Calculate dead neurons (per-neuron sparsity over batch)
        # Assuming batch is dim 0
        if t.dim() > 1:
            dead_ratio = (t.abs().sum(dim=0) < eps).sum().item() / t[0].numel()
        else:
            dead_ratio = sparsity

        # Calculate saturation ratio (for Sigmoid or Tanh typically)
        # We consider a value saturated if it's very close to 1.0 or -1.0
        saturation_ratio = (t.abs() > 0.95).sum().item() / numel

        # Standard statistics - compute directly on the detached tensor
        mean_val = t.mean().item()
        std_val = t.std().item() if numel > 1 else 0.0
        min_val = t.min().item()
        max_val = t.max().item()
        norm_val = t.norm().item()

        return {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "sparsity": sparsity,
            "dead_ratio": dead_ratio,
            "norm": norm_val,
            "saturation_ratio": saturation_ratio,
        }

    def _classify_activation_health(self, stats: Dict[str, float]) -> ActivationHealth:
        """Classify activation regime — delegates to engine when available."""
        if self._causal_engine is not None:
            return self._engine.activation.classify_health(stats)
        # Fallback heuristic when engine is not available
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

    def _detect_activation_shift(
        self, prev_stats: Dict[str, float], current_stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Detect activation shifts — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.activation.detect_shift(prev_stats, current_stats)

        # Fallback heuristic when engine is not available
        prev_health = self._classify_activation_health(prev_stats)
        current_health = self._classify_activation_health(current_stats)
        if prev_health != current_health:
            return {
                "type": f"{prev_health.value}_to_{current_health.value}",
                "confidence": 0.9,
            }
        return None

    def _classify_gradient_health(self, norm: float) -> GradientHealth:
        """Classify gradient health — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.gradient.classify_health(norm)
        # Fallback heuristic when engine is not available
        if norm < self.threshold_vanishing:
            return GradientHealth.VANISHING
        elif norm > self.threshold_exploding:
            return GradientHealth.EXPLODING
        elif norm < (self.threshold_vanishing * 100):
            return GradientHealth.SATURATED
        else:
            return GradientHealth.HEALTHY

    def _track_first_occurrence(self, failure_type: str, layer_name: str):
        """Track the first layer that encountered a specific failure."""
        if failure_type not in self.first_failure_step:
            self.first_failure_step[failure_type] = self.step
            self.first_failure_layer[failure_type] = layer_name

    def _detect_gradient_transition(
        self, prev_norm: float, current_norm: float
    ) -> Optional[Dict[str, Any]]:
        """Detect transitions in gradient health — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.gradient.detect_transition(prev_norm, current_norm)

        # Fallback heuristic when engine is not available
        prev_health = self._classify_gradient_health(prev_norm)
        current_health = self._classify_gradient_health(current_norm)

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

    def explain_failure(
        self, failure_type: str = "vanishing_gradients"
    ) -> List[CausalHypothesis]:
        """Provide ranked causal hypotheses — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.explain(failure_type)
        # Fallback: generate basic hypotheses from captured events
        hypotheses = []
        for event in self.events:
            if (
                event.event_type.value == failure_type
                or failure_type in str(event.event_type).lower()
            ):
                hypotheses.append(
                    CausalHypothesis(
                        description=(
                            f"{event.event_type.value} detected at "
                            f"{event.layer_name} (step {event.step})"
                        ),
                        confidence=event.confidence,
                        evidence=[event],
                        causal_chain=[f"{event.layer_name}@{event.step}"],
                    )
                )
        return hypotheses

    def _explain_vanishing_gradients(self) -> List[CausalHypothesis]:
        """Generate hypotheses for vanishing — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain._explain_vanishing_gradients()
        return self.explain_failure("vanishing_gradients")

    def _explain_exploding_gradients(self) -> List[CausalHypothesis]:
        """Generate hypotheses for exploding — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain._explain_exploding_gradients()
        return self.explain_failure("exploding_gradients")

    def _explain_dead_neurons(self) -> List[CausalHypothesis]:
        """Generate hypotheses for dead neurons — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain._explain_dead_neurons()
        return self.explain_failure("dead_neurons")

    def _explain_saturated_activations(self) -> List[CausalHypothesis]:
        """Generate hypotheses for saturated activations — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain._explain_saturated_activations()
        return self.explain_failure("saturated_activations")

    def get_causal_hypotheses(self) -> List[CausalHypothesis]:
        """Get all current causal hypotheses — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.get_causal_hypotheses()
        return []

    def trace_causal_chain(self, event_type: str) -> List[str]:
        """Trace the causal chain — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.trace_causal_chain(event_type)
        return []

    def detect_coupled_failures(self, window: int = 5) -> List[Dict[str, Any]]:
        """Detect coupled failures — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.coupling.detect(window)
        # Fallback: return empty list when engine not available
        return []

    def get_root_causes(self) -> List[CausalHypothesis]:
        """Identify root causes — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.get_root_causes()
        return []

    def _event_matches_failure_key(
        self,
        event: SemanticEvent,
        failure_key: str,
    ) -> bool:
        """Check event/failure key match — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.event_matches_failure_key(event, failure_key)
        return failure_key in str(event.event_type).lower()

    def _classify_data_health(
        self, tensor: torch.Tensor
    ) -> Tuple[DataHealth, Dict[str, Any]]:
        """Classify data health — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.data.classify_health(tensor)
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

    def _check_data_anomaly(self, tensor: torch.Tensor, layer_name: str):
        """Detect data anomalies — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.data.check_anomaly(tensor, layer_name)

        # Fallback implementation
        if not torch.is_floating_point(tensor):
            return

        current_health, health_metadata = self._classify_data_health(tensor)

        if current_health not in (DataHealth.NAN_DETECTED, DataHealth.INF_DETECTED):
            t = tensor.detach().float()
            if t.numel() == 0:
                current_mean = 0.0
                current_std = 0.0
            else:
                current_mean = t.mean().item()
                current_std = t.std().item() if t.numel() > 1 else 0.0

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

            self.previous_input_stats[layer_name] = {
                "mean": current_mean,
                "std": current_std,
            }

        prev_health = self.previous_data_health.get(layer_name, DataHealth.NORMAL)

        if current_health != prev_health:
            if current_health != DataHealth.NORMAL:
                self._track_first_occurrence(f"data_{current_health.value}", layer_name)
                if current_health in (
                    DataHealth.NAN_DETECTED,
                    DataHealth.INF_DETECTED,
                    DataHealth.DISTRIBUTION_SHIFT,
                ):
                    cache_path = self.disk_cache.save(
                        tensor, prefix=f"anomaly_{layer_name}"
                    )
                    health_metadata["tensor_cache_path"] = cache_path

            confidence = 1.0
            if current_health == DataHealth.DISTRIBUTION_SHIFT:
                mean_shift_val = health_metadata.get("mean_shift_sigma", 3.0)
                confidence = min(mean_shift_val * 0.2, 1.0)

            # Sample resources once per step
            resource_snapshot, resource_baseline = self._get_step_resource_snapshot(
                tensor.device
            )
            is_spike, spike_keys = self._is_memory_spike(
                resource_snapshot, resource_baseline
            )
            health_metadata["resources"] = resource_snapshot
            health_metadata["memory_spike"] = is_spike
            health_metadata["memory_spike_keys"] = spike_keys

            self.events.append(
                SemanticEvent(
                    event_type=EventType.DATA_ANOMALY,
                    layer_name=layer_name,
                    step=self.step,
                    from_state=prev_health.value,
                    to_state=current_health.value,
                    confidence=confidence,
                    metadata=health_metadata,
                )
            )
            self.previous_data_health[layer_name] = current_health

    def _explain_optimizer_instability(self) -> List[CausalHypothesis]:
        """Generate hypotheses for optimizer instability — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain._explain_optimizer_instability()
        return []

    def _explain_data_anomaly(self) -> List[CausalHypothesis]:
        """Generate hypotheses for data anomaly — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain._explain_data_anomaly()
        return []

    def _collapse_events(self) -> List[SemanticEvent]:
        """Collapse sequential events — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.collapse_events()
        return self.events

    def export_aquarium_package(self, package_path: str) -> str:
        """Export JSON package for Aquarium — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.export_aquarium_package(package_path)
        data = {
            "events": [
                {
                    "type": e.event_type.value,
                    "layer": e.layer_name,
                    "step": e.step,
                    "from": str(e.from_state),
                    "to": str(e.to_state),
                    "confidence": e.confidence,
                    "metadata": {
                        k: v
                        for k, v in e.metadata.items()
                        if isinstance(v, (str, int, float, bool, type(None)))
                    },
                }
                for e in self.events
            ],
            "hypotheses": [
                {
                    "description": h.description,
                    "confidence": h.confidence,
                    "causal_chain": h.causal_chain,
                }
                for h in self.get_causal_hypotheses()
            ],
            "couplings": self.detect_coupled_failures(),
            "first_failure_layer": dict(self.first_failure_layer),
            "first_failure_step": dict(self.first_failure_step),
            "loss_history": list(self.loss_history),
        }
        with open(package_path, "w") as f:
            json.dump(data, f, indent=2)
        return package_path

    def export_mermaid_causal_graph(self) -> str:
        """Export Mermaid causal graph — delegates to engine."""
        if self._causal_engine is not None:
            return self._engine.explain.export_mermaid_causal_graph()
        lines = ["graph TD"]
        for event in self.events:
            lines.append(f'    E_{event.id}["{event.layer_name} (step {event.step})"]')
        if len(self.events) > 1:
            for i in range(len(self.events) - 1):
                lines.append(f"    E_{self.events[i].id} --> E_{self.events[i + 1].id}")
        return "\n".join(lines)
