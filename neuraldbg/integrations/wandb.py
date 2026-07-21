"""
Weights & Biases Integration — NeuralDBG as a W&B Callback.

Add causal debugging to any W&B-tracked training run with one line:

    from neuraldbg.integrations.wandb import NeuralDBGCallback
    callback = NeuralDBGCallback(model, family="Transformer")
    # ... in training loop:
    callback.step(loss)

Events, causal chains, and root cause hypotheses are automatically logged
to your W&B run as structured tables and alerts.

Requirements:
    pip install wandb neuraldbg
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

# Optional import — fails gracefully if W&B not installed
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class NeuralDBGCallback:
    """W&B callback that adds causal debugging to any PyTorch training loop.

    Usage:
        callback = NeuralDBGCallback(model, family="CNN")
        with callback:
            for epoch in range(epochs):
                for batch in dataloader:
                    loss = train_step(model, batch)
                    loss.backward()
                    callback.step(loss)
                    optimizer.step()
                    # Optional: log every N steps
                    if step % 100 == 0:
                        callback.log_summary()

        # After training: get full report
        report = callback.report()
        print(report["summary"])
    """

    def __init__(
        self,
        model: nn.Module,
        family: str = "MLP",
        log_every_n_steps: int = 100,
        alert_on: Optional[List[str]] = None,
        strict_mode: bool = False,
        **dbg_kwargs,
    ):
        """
        Args:
            model: PyTorch model to monitor.
            family: Architecture family for calibrated thresholds
                    (MLP/CNN/RNN/Transformer/Hybrid/BlackSwan/RL).
            log_every_n_steps: How often to log summary tables to W&B.
            alert_on: Event types that trigger W&B alerts
                      (e.g. ["nan_detected", "gradient_health_transition"]).
            strict_mode: Higher thresholds = fewer false positives.
            **dbg_kwargs: Passed to NeuralDbg (threshold_vanishing, etc.).
        """
        if not HAS_WANDB:
            raise ImportError(
                "wandb is required for W&B integration. "
                "Install with: pip install wandb"
            )

        self.model = model
        self.family = family
        self.log_every_n_steps = log_every_n_steps
        self.alert_on = alert_on or [
            "nan_detected",
            "gradient_health_transition",
            "optimizer_instability",
        ]
        self.strict_mode = strict_mode

        self._dbg: Optional[NeuralDbg] = None
        self._step = 0
        self._losses: List[float] = []
        self._alerts_sent: set = set()
        self._dbg_kwargs = dbg_kwargs

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self):
        self._dbg = NeuralDbg(
            self.model,
            family=self.family,
            strict_mode=self.strict_mode,
            **self._dbg_kwargs,
        )
        self._dbg.__enter__()
        return self

    def __exit__(self, *args):
        if self._dbg:
            self._dbg.__exit__(*args)
        # Final summary
        self.log_summary()

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def step(self, loss: float):
        """Call after loss.backward(), before optimizer.step()."""
        self._step += 1
        self._losses.append(loss)

        if self._dbg is None:
            warnings.warn("NeuralDBGCallback used outside context manager. "
                          "Use 'with callback:' or call callback.start() first.")
            return

        self._dbg.record_loss(loss)
        self._dbg.step_iteration()

        # Periodic logging
        if self._step % self.log_every_n_steps == 0:
            self.log_summary()

        # Real-time alerts
        self._check_alerts()

    def start(self):
        """Start monitoring (alternative to context manager)."""
        self.__enter__()

    def stop(self):
        """Stop monitoring and log final summary."""
        self.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_summary(self):
        """Log current diagnostic summary to W&B."""
        if not HAS_WANDB or wandb.run is None:
            return

        events = self._get_events()
        chains = self._get_chains()
        summary = self._summarize(events, chains)

        # Log metrics
        wandb.log({
            "neuraldbg/total_events": len(events),
            "neuraldbg/anomaly_events": summary["anomaly_count"],
            "neuraldbg/causal_chains": len(chains),
            "neuraldbg/event_types": len(summary["event_types"]),
            "neuraldbg/step": self._step,
        }, step=self._step)

        # Log event type distribution as bar chart
        if summary["event_counts"]:
            wandb.log({
                "neuraldbg/event_distribution": wandb.plot.bar(
                    wandb.Table(
                        data=[[k, v] for k, v in summary["event_counts"].items()],
                        columns=["event_type", "count"],
                    ),
                    "event_type", "count",
                    title="Event Type Distribution",
                )
            }, step=self._step)

        # Log top causal chain as text
        if chains:
            top_chain = chains[0]
            root = getattr(top_chain, 'root_cause', str(top_chain))
            wandb.log({
                "neuraldbg/top_chain": str(root)[:200],
            }, step=self._step)

            # Log chains as table
            chain_data = []
            for i, chain in enumerate(chains[:10]):
                chain_data.append([
                    i + 1,
                    getattr(chain, 'root_cause', '?'),
                    getattr(chain, 'confidence', 0.0),
                    str(chain)[:300],
                ])
            if chain_data:
                wandb.log({
                    "neuraldbg/causal_chains_table": wandb.Table(
                        data=chain_data,
                        columns=["rank", "root_cause", "confidence", "chain"],
                    )
                }, step=self._step)

    def _check_alerts(self):
        """Send W&B alerts for critical events."""
        if not HAS_WANDB or wandb.run is None:
            return

        events = self._get_events()
        for event in events[-5:]:  # check recent events
            et = str(getattr(event, 'event_type', event.get('event_type', '')))
            if et in self.alert_on and et not in self._alerts_sent:
                detail = str(getattr(event, 'detail', event.get('detail', '')))[:200]
                wandb.alert(
                    title=f"NeuralDBG: {et}",
                    text=f"Step {self._step}: {detail}\n\n"
                         f"Run: {wandb.run.url}",
                    level=wandb.AlertLevel.WARN,
                )
                self._alerts_sent.add(et)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Return a full diagnostic report as a dictionary."""
        events = self._get_events()
        chains = self._get_chains()
        summary = self._summarize(events, chains)

        return {
            "summary": summary["summary_text"],
            "total_events": len(events),
            "anomaly_events": summary["anomaly_count"],
            "event_types": sorted(summary["event_types"]),
            "event_counts": summary["event_counts"],
            "causal_chains": [
                {
                    "root_cause": getattr(c, 'root_cause', str(c)),
                    "confidence": getattr(c, 'confidence', 0.0),
                    "chain": str(c)[:300],
                }
                for c in chains[:10]
            ],
            "warnings": summary["warnings"],
            "recommendations": summary["recommendations"],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_events(self) -> List:
        if self._dbg is None:
            return []
        try:
            return self._dbg.dump_events()
        except Exception:
            return []

    def _get_chains(self) -> List:
        if self._dbg is None:
            return []
        try:
            return self._dbg.explain_causal()
        except Exception:
            return []

    def _summarize(self, events, chains) -> Dict:
        event_types = set()
        event_counts: Dict[str, int] = {}
        anomaly_count = 0

        anomaly_keywords = {"vanishing", "exploding", "dead", "saturated",
                           "nan_detected", "anomalous", "instability",
                           "data_anomaly", "silent_corruption"}

        for e in events:
            et = str(getattr(e, 'event_type', e.get('event_type', '')))
            ts = str(getattr(e, 'to_state', e.get('to_state', ''))).lower()
            event_types.add(et)
            event_counts[et] = event_counts.get(et, 0) + 1

            # Count anomalies
            if any(kw in et.lower() or kw in ts for kw in anomaly_keywords):
                anomaly_count += 1

        # Build summary text
        warnings_list = []
        recommendations = []

        if anomaly_count == 0:
            summary_text = (
                f"NeuralDBG: Training healthy after {self._step} steps. "
                f"{len(events)} total events, 0 anomalies."
            )
        elif anomaly_count <= 5:
            summary_text = (
                f"NeuralDBG: Mild anomalies detected ({anomaly_count} events). "
                f"Check W&B dashboard for event distribution."
            )
        else:
            summary_text = (
                f"NeuralDBG: ⚠ {anomaly_count} anomalies detected across "
                f"{len(event_types)} event types. "
                f"{len(chains)} causal chains traced. "
                f"Review 'neuraldbg/causal_chains_table' in W&B."
            )
            recommendations.append("Review top causal chains for root cause.")

        # Specific recommendations based on event types
        if any("exploding" in et.lower() for et in event_types):
            recommendations.append(
                "Gradient explosion detected. Reduce learning rate or add "
                "gradient clipping (max_norm=1.0)."
            )
        if any("vanishing" in et.lower() for et in event_types):
            recommendations.append(
                "Vanishing gradients detected. Check activation functions "
                "(avoid Sigmoid), add BatchNorm, or increase LR."
            )
        if any("nan" in et.lower() for et in event_types):
            recommendations.append(
                "NaN detected. Check data pipeline for NaN/Inf values, "
                "verify loss function domain (add epsilon to log/sqrt)."
            )
        if any("dead" in et.lower() or "saturated" in et.lower() for et in event_types):
            recommendations.append(
                "Dead/saturated neurons detected. Check weight initialization "
                "and learning rate. Consider using LeakyReLU."
            )

        return {
            "summary_text": summary_text,
            "anomaly_count": anomaly_count,
            "event_types": event_types,
            "event_counts": event_counts,
            "warnings": warnings_list,
            "recommendations": recommendations,
        }


# ------------------------------------------------------------------
# Convenience: auto-patch W&B init
# ------------------------------------------------------------------

def patch_wandb_init(family: str = "MLP", **kwargs):
    """Monkey-patch wandb.init to auto-instrument with NeuralDBG.

    Usage:
        from neuraldbg.integrations.wandb import patch_wandb_init
        patch_wandb_init(family="Transformer")

        # Now all wandb.init() calls get NeuralDBG automatically
        wandb.init(project="my-project")
        # ... train as usual, events logged automatically

    Note: This is a convenience for existing codebases. For new projects,
    use the NeuralDBGCallback directly.
    """
    if not HAS_WANDB:
        raise ImportError("wandb required. Install with: pip install wandb")

    import wandb as _wandb
    _original_init = _wandb.init

    def _patched_init(*args, **kwargs):
        run = _original_init(*args, **kwargs)
        # Store callback on run object for later retrieval
        run._neuraldbg_family = family
        run._neuraldbg_kwargs = kwargs
        return run

    _wandb.init = _patched_init
