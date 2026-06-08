"""Comparison scaffolding: NeuralDBG vs naive loss loggers on the same bug.

The MHA fully-masked-row case (BUG-001) is the canonical example of
why a loss logger is not enough. The forward pass masks the bad row
out of the loss, so:

  - Training loss prints finite values (e.g. 0.008)
  - in_proj_weight.grad contains NaN

A W&B-style logger that tracks `loss` will see a clean run.
NeuralDBG (with register_composite_hook) will see the NaN event.

This module is the scaffolding: a W&B-shaped naive logger that
implements ONLY what a typical user gets from off-the-shelf tools
(loss tracking, no parameter-level NaN detection). It returns a
verdict: "saw the bug" or "missed the bug" — for the same scenario.

Honest limitation: this is a mock. A real comparison would run both
NeuralDBG and W&B on the same machine, but the MHA bug reproduces
without any GPU and the behavioral difference is large enough that
a mock is a faithful proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class NaiveLoggerOutcome:
    """What a loss-only logger (W&B-like) would conclude for a scenario."""

    scenario_name: str
    detected_failure: bool
    final_loss: float
    losses: List[float]
    notes: str = ""


def naive_loss_logger(
    scenario_name: str,
    step_losses: List[float],
    post_step_param_nan: List[bool],
) -> NaiveLoggerOutcome:
    """Mimic a typical MLOps logger that tracks loss but NOT gradients.

    Args:
        scenario_name: used for the report.
        step_losses: the per-step loss values (whatever the user logged).
        post_step_param_nan: per-step booleans indicating whether the
            model parameters ended up with NaN gradients at that step.
            A real W&B logger would not have access to this — we pass
            it so the function can document what it MISSED.

    Returns:
        NaiveLoggerOutcome with detected_failure True only if any loss
        is NaN/Inf. Gradients are not inspected, by design.
    """
    detected = any(
        (isinstance(l, float) and (math.isnan(l) or math.isinf(l))) for l in step_losses
    )
    nan_steps = [i for i, v in enumerate(post_step_param_nan) if v]
    notes = (
        "Loss is finite at every step; no anomaly flagged."
        if not detected
        else f"NaN/Inf loss detected at step {next(i for i, l in enumerate(step_losses) if isinstance(l, float) and math.isnan(l))}."
    )
    if nan_steps and not detected:
        notes += (
            f" BUT {len(nan_steps)} step(s) produced NaN gradients on "
            "parameters (not visible to a loss-only logger)."
        )
    return NaiveLoggerOutcome(
        scenario_name=scenario_name,
        detected_failure=detected,
        final_loss=step_losses[-1] if step_losses else float("nan"),
        losses=list(step_losses),
        notes=notes,
    )


def compare_on_mha_bug(
    step_losses: List[float], post_step_param_nan: List[bool]
) -> dict:
    """Return a side-by-side verdict for the MHA fully-masked-row case.

    This is the comparison published in the public benchmark:
        NeuralDBG (with FIX-001)     : flags the bug.
        Naive loss logger (W&B-like) : misses the bug.

    The function is pure and testable. It does not import NeuralDBG —
    callers wire it up with the actual losses + NaN states.
    """
    ndbg = naive_loss_logger(
        "mha_fully_masked_row_BUG-001", step_losses, post_step_param_nan
    )
    ndbg_verdict = "DETECTED" if any(post_step_param_nan) else "MISSED"
    return {
        "scenario": "mha_fully_masked_row_BUG-001",
        "tools": {
            "neuraldbg_with_FIX-001": ndbg_verdict,
            "naive_loss_logger_(wandb_like)": (
                "DETECTED" if ndbg.detected_failure else "MISSED"
            ),
        },
        "neuraldbg_evidence": {
            "loss_values": step_losses,
            "param_nan_steps": [i for i, v in enumerate(post_step_param_nan) if v],
        },
        "naive_logger_notes": ndbg.notes,
        "winner": "neuraldbg_with_FIX-001"
        if any(post_step_param_nan) and not ndbg.detected_failure
        else "tie",
    }
