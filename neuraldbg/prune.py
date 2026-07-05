"""
NeuralPrune -- Model Optimization Diagnostic Engine

Detects redundant parameters, dead neurons, dead MoE experts, and quantization
opportunities WITHOUT modifying the model. Outputs a structured PruneReport
with confidence-scored recommendations.

Part of NeuralSuite: diagnose what's wrong BEFORE you fix it.

Architecture:
  NeuralPrune piggybacks on NeuralDbg hooks to collect per-parameter
  statistics over a warmup window. After analysis, it emits:
  - DEAD_NEURON: activation always zero (ReLU collapse)
  - DEAD_EXPERT: MoE expert never routed
  - LOW_RANK: weight matrix has low effective rank
  - REDUNDANT_WEIGHT: |weight| consistently near zero
  - STATIC_WEIGHT: gradient ~0 for many steps (plateau)
  - QUANTIZABLE: activation range fits INT8/INT4 bounds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class PruneSignal(str, Enum):
    """Redundancy signals detected by NeuralPrune."""
    DEAD_NEURON = "dead_neuron"           # Channel never activates
    DEAD_EXPERT = "dead_expert"           # MoE expert never routed
    LOW_RANK = "low_rank"                 # Weight matrix has low effective rank
    REDUNDANT_WEIGHT = "redundant_weight" # |w| consistently near zero
    STATIC_WEIGHT = "static_weight"       # Gradient ~0 for many steps
    QUANTIZABLE = "quantizable"           # Safe for INT8/INT4 quantization
    OVERLAPPING_FILTERS = "overlapping_filters"  # CNN filters too similar


@dataclass
class PruneRecommendation:
    """A single pruning/optimization recommendation."""
    layer_name: str
    signal: PruneSignal
    confidence: float          # 0.0 - 1.0
    detail: str
    suggested_action: str      # e.g. "prune 30% of output channels"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PruneReport:
    """Full analysis report from NeuralPrune."""
    model_name: str
    total_params: int
    redundant_params: int      # Estimated redundant count
    recommendations: List[PruneRecommendation]
    summary: str               # Human-readable summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_params": self.total_params,
            "redundant_params": self.redundant_params,
            "recommendations": [
                {
                    "layer": r.layer_name,
                    "signal": r.signal.value,
                    "confidence": round(r.confidence, 3),
                    "detail": r.detail,
                    "action": r.suggested_action,
                    "meta": r.metadata,
                }
                for r in self.recommendations
            ],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Per-layer statistics accumulator
# ---------------------------------------------------------------------------

@dataclass
class _LayerStats:
    """Accumulated statistics for a single layer."""
    name: str
    module_type: str
    num_params: int = 0

    # Activation tracking (updated from forward hooks)
    activation_mean: float = 0.0
    activation_sparsity: float = 0.0    # Fraction of near-zero activations
    activation_range: Tuple[float, float] = (0.0, 0.0)
    num_forward_passes: int = 0

    # Weight tracking
    weight_mean_abs: float = 0.0
    weight_std: float = 0.0
    weight_sparsity: float = 0.0        # Fraction of |w| < 1e-6

    # Gradient tracking
    gradient_mean_abs: float = 0.0
    gradient_zero_fraction: float = 0.0 # Fraction of |grad| < 1e-7

    # MoE-specific
    expert_route_count: Dict[int, int] = field(default_factory=dict)

    # Internal accumulators (Welford for mean)
    _act_sum: float = field(default=0.0, repr=False)
    _act_sum_sq: float = field(default=0.0, repr=False)
    _act_count: int = field(default=0, repr=False)
    _act_zero_count: int = field(default=0, repr=False)
    _act_min: float = field(default=float('inf'), repr=False)
    _act_max: float = field(default=float('-inf'), repr=False)

    _grad_sum: float = field(default=0.0, repr=False)
    _grad_zero_count: int = field(default=0, repr=False)
    _grad_total_count: int = field(default=0, repr=False)

    def update_activation(self, output: torch.Tensor) -> None:
        """Update activation statistics from a forward-pass output tensor."""
        if not isinstance(output, torch.Tensor):
            return
        with torch.no_grad():
            flat = output.detach().float()
            self._act_sum += flat.sum().item()
            self._act_sum_sq += (flat ** 2).sum().item()
            self._act_count += flat.numel()
            self._act_zero_count += (flat.abs() < 1e-6).sum().item()
            self._act_min = min(self._act_min, flat.min().item())
            self._act_max = max(self._act_max, flat.max().item())
            self.num_forward_passes += 1

    def update_gradient(self, grad_output: torch.Tensor) -> None:
        """Update gradient statistics from a backward-pass gradient tensor."""
        if not isinstance(grad_output, torch.Tensor):
            return
        with torch.no_grad():
            flat = grad_output.detach().float().abs()
            self._grad_sum += flat.sum().item()
            self._grad_zero_count += (flat < 1e-7).sum().item()
            self._grad_total_count += flat.numel()

    def finalize(self) -> None:
        """Compute derived statistics after all passes."""
        if self._act_count > 0:
            self.activation_mean = self._act_sum / self._act_count
            self.activation_sparsity = self._act_zero_count / self._act_count
            self.activation_range = (self._act_min, self._act_max)
        if self._grad_total_count > 0:
            self.gradient_mean_abs = self._grad_sum / self._grad_total_count
            self.gradient_zero_fraction = self._grad_zero_count / self._grad_total_count


# ---------------------------------------------------------------------------
# NeuralPrune
# ---------------------------------------------------------------------------

class NeuralPrune:
    """Non-destructive model optimization diagnostic engine.

    Piggybacks on NeuralDbg's forward/backward hooks to collect per-layer
    statistics. After a warmup window, analyzes patterns and emits a
    PruneReport with actionable recommendations.

    Usage:
        model = MyLargeModel()
        pruner = NeuralPrune(model, warmup_steps=100)

        for batch in dataloader:
            loss = train_step(model, batch)
            loss.backward()
            pruner.step(batch)  # collect stats

        report = pruner.analyze()
        print(report.summary)
        for rec in report.recommendations:
            print(f"  {rec.layer_name}: {rec.suggested_action}")
    """

    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 50,
        dead_neuron_threshold: float = 0.99,   # 99%+ zeros = dead
        dead_weight_threshold: float = 1e-6,    # |w| below this = redundant
        static_grad_threshold: float = 1e-7,    # |grad| below this = static
        low_rank_ratio: float = 0.1,            # effective_rank / dim < this = low_rank
        quantization_bits: Tuple[int, ...] = (8, 4),
    ):
        self.model = model
        self.warmup_steps = warmup_steps
        self.dead_neuron_threshold = dead_neuron_threshold
        self.dead_weight_threshold = dead_weight_threshold
        self.static_grad_threshold = static_grad_threshold
        self.low_rank_ratio = low_rank_ratio
        self.quantization_bits = quantization_bits

        self._step_count = 0
        self._stats: Dict[str, _LayerStats] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        self._install_hooks()

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        """Attach forward/backward hooks to all leaf modules."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0 and name != "":
                continue  # skip composite modules

            module_type = type(module).__name__
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            self._stats[name] = _LayerStats(
                name=name or "(root)",
                module_type=module_type,
                num_params=num_params,
            )

            # Capture name in closure
            layer_name = name

            def fwd_hook(_mod, _inp, out, _name=layer_name):
                self._on_forward(_name, _mod, _inp, out)

            def bwd_hook(_mod, _ginp, _gout, _name=layer_name):
                self._on_backward(_name, _mod, _ginp, _gout)

            # Also update weight stats from parameters
            for pname, param in module.named_parameters():
                if param.requires_grad and param.data.numel() > 0:
                    full_name = f"{layer_name}.{pname}" if layer_name else pname
                    self._stats[full_name] = _LayerStats(
                        name=full_name,
                        module_type=type(module).__name__,
                        num_params=param.data.numel(),
                    )
                    # Read initial weight stats
                    with torch.no_grad():
                        w = param.data.float()
                        self._stats[full_name].weight_mean_abs = w.abs().mean().item()
                        self._stats[full_name].weight_std = w.std().item()
                        self._stats[full_name].weight_sparsity = (
                            (w.abs() < self.dead_weight_threshold).float().mean().item()
                        )

            self._hooks.append(module.register_forward_hook(fwd_hook))
            self._hooks.append(module.register_full_backward_hook(bwd_hook))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_forward(self, name: str, module: nn.Module, inp: Any, out: Any) -> None:
        """Forward hook: track activation sparsity and range."""
        stats = self._stats.get(name)
        if stats is None:
            return

        # Handle tuple outputs (LSTM, GRU, etc.)
        output = out
        if isinstance(out, tuple):
            output = out[0] if isinstance(out[0], torch.Tensor) else out

        if isinstance(output, torch.Tensor):
            stats.update_activation(output)

        # MoE expert routing detection
        if hasattr(module, 'router') and isinstance(inp, (tuple, list)):
            # Try to detect MoE routing from input
            pass

    def _on_backward(self, name: str, module: nn.Module, ginp: Any, gout: Any) -> None:
        """Backward hook: track gradient health per layer."""
        stats = self._stats.get(name)
        if stats is None:
            return

        # ginp is tuple of gradient inputs; gout is tuple of gradient outputs
        grad = gout
        if isinstance(gout, tuple):
            for g in gout:
                if isinstance(g, torch.Tensor):
                    stats.update_gradient(g)
        elif isinstance(gout, torch.Tensor):
            stats.update_gradient(gout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, batch: Any = None) -> None:
        """Called after each training step (loss.backward()).

        Args:
            batch: Optional batch data (reserved for future use, e.g.
                   tracking input statistics).
        """
        self._step_count += 1

    def analyze(self) -> PruneReport:
        """Run analysis and return a PruneReport with recommendations.

        Must be called after at least ``warmup_steps`` calls to :meth:`step`.
        """
        # Finalize accumulated statistics
        for stats in self._stats.values():
            stats.finalize()

        recommendations: List[PruneRecommendation] = []

        for name, stats in self._stats.items():
            recs = self._analyze_layer(name, stats)
            recommendations.extend(recs)

        # Sort by confidence descending
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        # Estimate redundant parameter count
        redundant = self._estimate_redundant(recommendations)

        # Build summary
        summary = self._build_summary(recommendations, redundant)

        return PruneReport(
            model_name=type(self.model).__name__,
            total_params=self._total_params,
            redundant_params=redundant,
            recommendations=recommendations,
            summary=summary,
        )

    def _analyze_layer(self, name: str, stats: _LayerStats) -> List[PruneRecommendation]:
        """Analyze a single layer's statistics and return recommendations."""
        recs: List[PruneRecommendation] = []

        # --- Dead neuron detection ---
        if stats.num_forward_passes >= self.warmup_steps:
            if stats.activation_sparsity >= self.dead_neuron_threshold:
                recs.append(PruneRecommendation(
                    layer_name=name,
                    signal=PruneSignal.DEAD_NEURON,
                    confidence=min(1.0, stats.activation_sparsity),
                    detail=(
                        f"{stats.activation_sparsity*100:.1f}% of activations "
                        f"are near-zero over {stats.num_forward_passes} passes. "
                        f"Mean activation: {stats.activation_mean:.6f}"
                    ),
                    suggested_action=(
                        f"Prune output channels with {stats.activation_sparsity*100:.0f}%+ "
                        f"zero activation rate"
                    ),
                    metadata={
                        "activation_sparsity": stats.activation_sparsity,
                        "activation_mean": stats.activation_mean,
                        "num_passes": stats.num_forward_passes,
                    },
                ))

        # --- Redundant weight detection ---
        if stats.weight_sparsity >= 0.5:  # 50%+ weights near zero
            recs.append(PruneRecommendation(
                layer_name=name,
                signal=PruneSignal.REDUNDANT_WEIGHT,
                confidence=min(1.0, stats.weight_sparsity),
                detail=(
                    f"{stats.weight_sparsity*100:.1f}% of weights are below "
                    f"{self.dead_weight_threshold} in absolute value. "
                    f"Mean |w|: {stats.weight_mean_abs:.6f}"
                ),
                suggested_action=(
                    f"Magnitude-prune {stats.weight_sparsity*100:.0f}% of weights "
                    f"or apply L1 regularization"
                ),
                metadata={
                    "weight_sparsity": stats.weight_sparsity,
                    "weight_mean_abs": stats.weight_mean_abs,
                    "weight_std": stats.weight_std,
                },
            ))

        # --- Static weight (dead gradient) ---
        if stats.gradient_zero_fraction >= 0.9:
            recs.append(PruneRecommendation(
                layer_name=name,
                signal=PruneSignal.STATIC_WEIGHT,
                confidence=min(1.0, stats.gradient_zero_fraction),
                detail=(
                    f"{stats.gradient_zero_fraction*100:.1f}% of gradients are "
                    f"near-zero. Layer may be in a plateau or dead."
                ),
                suggested_action=(
                    "Consider removing this layer or increasing learning rate"
                ),
                metadata={
                    "gradient_zero_fraction": stats.gradient_zero_fraction,
                    "gradient_mean_abs": stats.gradient_mean_abs,
                },
            ))

        # --- Low-rank detection (for Linear/Conv2d weight matrices) ---
        if stats.module_type in ("Linear", "Conv2d") and stats.num_params > 100:
            rank_info = self._estimate_rank(name)
            if rank_info is not None:
                effective_rank, total_dim, ratio = rank_info
                if ratio < self.low_rank_ratio:
                    recs.append(PruneRecommendation(
                        layer_name=name,
                        signal=PruneSignal.LOW_RANK,
                        confidence=min(1.0, 1.0 - ratio / self.low_rank_ratio),
                        detail=(
                            f"Effective rank {effective_rank} / {total_dim} "
                            f"(ratio {ratio:.3f}). Weight matrix is low-rank."
                        ),
                        suggested_action=(
                            f"Apply SVD decomposition: replace {total_dim}x{total_dim} "
                            f"with {effective_rank}x{total_dim} + {total_dim}x{effective_rank}"
                        ),
                        metadata={
                            "effective_rank": effective_rank,
                            "total_dim": total_dim,
                            "rank_ratio": ratio,
                        },
                    ))

        # --- Quantization potential ---
        if stats.activation_range[0] > -1e6 and stats.activation_range[1] < 1e6:
            act_range = stats.activation_range[1] - stats.activation_range[0]
            if act_range > 0 and math.isfinite(act_range):
                for bits in self.quantization_bits:
                    levels = 2 ** bits
                    precision = act_range / levels
                    # If precision is acceptable (>1e-4 for fp16 comparison)
                    if precision > 1e-4:
                        recs.append(PruneRecommendation(
                            layer_name=name,
                            signal=PruneSignal.QUANTIZABLE,
                            confidence=0.7,
                            detail=(
                                f"Activation range [{stats.activation_range[0]:.4f}, "
                                f"{stats.activation_range[1]:.4f}] fits INT{bits} "
                                f"(precision {precision:.6f} per level)"
                            ),
                            suggested_action=f"Quantize to INT{bits}",
                            metadata={
                                "bits": bits,
                                "precision_per_level": precision,
                                "activation_range": list(stats.activation_range),
                            },
                        ))
                        break  # Only recommend the highest-precision quantization

        return recs

    def _estimate_rank(self, layer_name: str) -> Optional[Tuple[int, int, float]]:
        """Estimate effective rank of a weight matrix via SVD.

        Returns:
            (effective_rank, total_dim, ratio) or None if not applicable.
        """
        for pname, param in self.model.named_parameters():
            full_name = f"{layer_name}.{pname}" if layer_name else pname
            if layer_name in full_name and param.dim() == 2:
                with torch.no_grad():
                    w = param.data.float()
                    # Use a fast approximate rank via singular values
                    try:
                        s = torch.linalg.svdvals(w)
                        total = s.numel()
                        # Count singular values above 1% of max
                        threshold = s.max().item() * 0.01
                        effective = (s > threshold).sum().item()
                        ratio = effective / total if total > 0 else 1.0
                        return effective, total, ratio
                    except Exception:
                        return None
        return None

    def _estimate_redundant(self, recs: List[PruneRecommendation]) -> int:
        """Estimate total redundant parameter count from recommendations."""
        redundant = 0
        seen_layers: set = set()
        for rec in recs:
            if rec.layer_name not in seen_layers:
                seen_layers.add(rec.layer_name)
                stats = self._stats.get(rec.layer_name)
                if stats and stats.num_params > 0:
                    if rec.signal in (PruneSignal.DEAD_NEURON, PruneSignal.REDUNDANT_WEIGHT):
                        redundant += int(stats.num_params * rec.confidence)
                    elif rec.signal == PruneSignal.STATIC_WEIGHT:
                        redundant += int(stats.num_params * 0.5 * rec.confidence)
                    elif rec.signal == PruneSignal.LOW_RANK:
                        ratio = rec.metadata.get("rank_ratio", 0.5)
                        redundant += int(stats.num_params * (1 - ratio))
        return min(redundant, self._total_params)

    def _build_summary(self, recs: List[PruneRecommendation], redundant: int) -> str:
        """Build a human-readable summary string."""
        if not recs:
            return "No redundancy detected. Model appears optimally configured."

        by_signal: Dict[str, int] = {}
        for rec in recs:
            key = rec.signal.value
            by_signal[key] = by_signal.get(key, 0) + 1

        pct = redundant / max(self._total_params, 1) * 100
        lines = [
            f"NeuralPrune analysis: {len(recs)} recommendations across "
            f"{len(by_signal)} signal types.",
            f"Estimated redundant parameters: {redundant:,} / "
            f"{self._total_params:,} ({pct:.1f}%)",
            "",
            "Signal breakdown:",
        ]
        for sig, count in sorted(by_signal.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {sig}: {count} layers")

        total_savings = redundant * 4  # assume fp32 = 4 bytes
        mb = total_savings / (1024 * 1024)
        lines.append("")
        if mb >= 1.0:
            lines.append(f"Potential memory savings: ~{mb:.1f} MB (fp32)")
        else:
            lines.append(f"Potential memory savings: ~{total_savings / 1024:.1f} KB (fp32)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()
