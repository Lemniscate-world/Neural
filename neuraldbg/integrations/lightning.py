"""
PyTorch Lightning Integration — NeuralDBG as a Lightning Callback.

Add causal debugging to any Lightning training run with one callback:

    from neuraldbg.integrations.lightning import NeuralDBGLightningCallback
    trainer = pl.Trainer(callbacks=[NeuralDBGLightningCallback(family="CNN")])

Events and causal chains are logged to the Lightning logger (W&B, TensorBoard,
or CSV) automatically.

Requirements:
    pip install pytorch-lightning neuraldbg
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neuraldbg import NeuralDbg

# Optional import
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    Callback = object  # type: ignore


if HAS_LIGHTNING:

    class NeuralDBGLightningCallback(Callback):
        """Lightning callback for NeuralDBG causal monitoring.

        Usage:
            trainer = pl.Trainer(
                callbacks=[NeuralDBGLightningCallback(family="Transformer")]
            )
            trainer.fit(model, dataloader)
        """

        def __init__(
            self,
            family: str = "MLP",
            log_every_n_steps: int = 100,
            strict_mode: bool = False,
            **dbg_kwargs,
        ):
            """
            Args:
                family: Architecture family for calibrated thresholds.
                log_every_n_steps: How often to log summary.
                strict_mode: Higher thresholds = fewer false positives.
            """
            super().__init__()
            self.family = family
            self.log_every_n_steps = log_every_n_steps
            self.strict_mode = strict_mode
            self._dbg_kwargs = dbg_kwargs

            self._dbg: Optional[NeuralDbg] = None
            self._step = 0

        def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
            """Initialize NeuralDBG when training starts."""
            self._dbg = NeuralDbg(
                pl_module,
                family=self.family,
                strict_mode=self.strict_mode,
                **self._dbg_kwargs,
            ).__enter__()

        def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
            """Clean up and log final summary."""
            if self._dbg:
                self._log_summary(trainer)
                self._dbg.__exit__(None, None, None)
                self._dbg = None

        def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Any,
            batch: Any,
            batch_idx: int,
        ):
            """Called after loss.backward() by Lightning."""
            self._step += 1

            if self._dbg is None:
                return

            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if isinstance(loss, torch.Tensor):
                loss_val = loss.item()
            else:
                loss_val = float(loss) if loss is not None else 0.0

            self._dbg.record_loss(loss_val)
            self._dbg.step_iteration()

            # Periodic logging
            if self._step % self.log_every_n_steps == 0:
                self._log_summary(trainer)

        def _log_summary(self, trainer: "pl.Trainer"):
            """Log diagnostic summary to Lightning's logger."""
            if self._dbg is None:
                return

            try:
                events = self._dbg.dump_events()
                chains = self._dbg.explain_causal()
            except Exception:
                return

            anomaly_count = sum(
                1 for e in events
                if any(kw in str(e.get("event_type", "")).lower()
                       for kw in ("vanishing", "exploding", "dead", "nan",
                                  "anomalous", "instability"))
            )

            event_types = list(set(e.get("event_type", "?") for e in events))

            # Log to all Lightning loggers
            metrics = {
                "neuraldbg/total_events": len(events),
                "neuraldbg/anomaly_events": anomaly_count,
                "neuraldbg/causal_chains": len(chains),
                "neuraldbg/event_types": len(event_types),
                "neuraldbg/step": self._step,
            }
            trainer.logger.log_metrics(metrics, step=self._step)

            # Log top chain as text
            if chains:
                top = str(chains[0])[:300]
                trainer.logger.log_metrics(
                    {"neuraldbg/top_chain": top},
                    step=self._step,
                )


else:
    # Graceful fallback: class that raises helpful error on instantiation
    class NeuralDBGLightningCallback:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pytorch_lightning is required for Lightning integration. "
                "Install with: pip install pytorch-lightning"
            )
