"""Real tool comparison: NeuralDBG vs W&B / MLflow / TensorBoard.

Runs all 4 benchmark scenarios with each tool instrumenting the same
training loop.  For W&B, MLflow, and TensorBoard only scalar loss is
logged (the typical off-the-shelf usage).  After each run the script
checks what the tool COULD have seen and scores detection / localization.

Output: benchmark_public/comparison_results.json + stdout table.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from benchmark_public.scenarios import PUBLIC_SCENARIOS
from benchmark_public.run import run_scenario, evaluate

SEED = 42


# ---------------------------------------------------------------------------
# Per-tool training loops (same model, same data, same bug injector)
# ---------------------------------------------------------------------------


def _train_neuraldbg(scenario):
    dbg, gt = run_scenario(scenario)
    hyps = dbg.explain_failure()
    return {
        "losses": list(dbg.loss_history),
        "hypotheses": [{"desc": h.description, "chain": h.causal_chain} for h in hyps],
    }


def _train_wandb(scenario):
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    import wandb

    tmpdir = tempfile.mkdtemp(prefix="bench_wb_")
    try:
        wandb.init(
            project="neuralsuite-comparison",
            dir=tmpdir,
            settings=wandb.Settings(silent=True),
        )
        torch.manual_seed(SEED)
        model = scenario.model_builder()
        data = scenario.data_builder()
        losses = []
        nan_grad_steps = []
        has_nan = False

        if scenario.step_fn is not None:
            for step in range(scenario.num_steps):
                scenario.bug_injector(model, step)
                x_t, am, kpm = data[0], data[1], data[2]
                out, _ = model(x_t, x_t, x_t, attn_mask=am, key_padding_mask=kpm)
                loss = out[:2, :].sum()
                loss.backward()
                lv = loss.item()
                losses.append(lv)
                wandb.log({"loss": lv, "step": step})
                step_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any().item()
                    for p in model.parameters()
                )
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                model.zero_grad()
        else:
            x, y = data
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for step in range(scenario.num_steps):
                optimizer.zero_grad()
                scenario.bug_injector(model, step)
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                lv = loss.item()
                losses.append(lv)
                wandb.log({"loss": lv, "step": step})
                step_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any().item()
                    for p in model.parameters()
                )
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                optimizer.step()

        wandb.finish()
        return {
            "losses": losses,
            "has_nan_grads": has_nan,
            "nan_grad_steps": nan_grad_steps,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _train_mlflow(scenario):
    import mlflow

    tmpdir = tempfile.mkdtemp(prefix="bench_ml_")
    uri = Path(tmpdir).as_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("neuralsuite-comparison")
        with mlflow.start_run():
            torch.manual_seed(SEED)
            model = scenario.model_builder()
            data = scenario.data_builder()
            losses = []
            nan_grad_steps = []
            has_nan = False

            if scenario.step_fn is not None:
                for step in range(scenario.num_steps):
                    scenario.bug_injector(model, step)
                    x_t, am, kpm = data[0], data[1], data[2]
                    out, _ = model(x_t, x_t, x_t, attn_mask=am, key_padding_mask=kpm)
                    loss = out[:2, :].sum()
                    loss.backward()
                    lv = loss.item()
                    losses.append(lv)
                    mlflow.log_metric("loss", lv, step=step)
                    step_nan = any(
                        p.grad is not None and torch.isnan(p.grad).any().item()
                        for p in model.parameters()
                    )
                    if step_nan:
                        nan_grad_steps.append(step)
                        has_nan = True
                    model.zero_grad()
            else:
                x, y = data
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                for step in range(scenario.num_steps):
                    optimizer.zero_grad()
                    scenario.bug_injector(model, step)
                    out = model(x)
                    loss = nn.CrossEntropyLoss()(out, y)
                    loss.backward()
                    lv = loss.item()
                    losses.append(lv)
                    mlflow.log_metric("loss", lv, step=step)
                    step_nan = any(
                        p.grad is not None and torch.isnan(p.grad).any().item()
                        for p in model.parameters()
                    )
                    if step_nan:
                        nan_grad_steps.append(step)
                        has_nan = True
                    optimizer.step()

        return {
            "losses": losses,
            "has_nan_grads": has_nan,
            "nan_grad_steps": nan_grad_steps,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _train_tensorboard(scenario):
    from torch.utils.tensorboard import SummaryWriter

    tmpdir = tempfile.mkdtemp(prefix="bench_tb_")
    try:
        writer = SummaryWriter(log_dir=tmpdir)
        torch.manual_seed(SEED)
        model = scenario.model_builder()
        data = scenario.data_builder()
        losses = []
        nan_grad_steps = []
        has_nan = False

        if scenario.step_fn is not None:
            for step in range(scenario.num_steps):
                scenario.bug_injector(model, step)
                x_t, am, kpm = data[0], data[1], data[2]
                out, _ = model(x_t, x_t, x_t, attn_mask=am, key_padding_mask=kpm)
                loss = out[:2, :].sum()
                loss.backward()
                lv = loss.item()
                losses.append(lv)
                writer.add_scalar("loss", lv, step)
                step_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any().item()
                    for p in model.parameters()
                )
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                model.zero_grad()
        else:
            x, y = data
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            for step in range(scenario.num_steps):
                optimizer.zero_grad()
                scenario.bug_injector(model, step)
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                lv = loss.item()
                losses.append(lv)
                writer.add_scalar("loss", lv, step)
                step_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any().item()
                    for p in model.parameters()
                )
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                optimizer.step()

        writer.close()
        return {
            "losses": losses,
            "has_nan_grads": has_nan,
            "nan_grad_steps": nan_grad_steps,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def _has_nan_loss(losses):
    return any(
        isinstance(l, float) and (math.isnan(l) or math.isinf(l)) for l in losses
    )


def _score_loss_logger(result, scenario):
    """Score a loss-only tool (W&B / MLflow / TensorBoard).

    Detection: 1.0 if loss is NaN/Inf or there is a visible spike >5x baseline.
    Localization: always 0.0 (loss-only tools cannot name the layer).
    """
    losses = result["losses"]
    nan_grads = result.get("has_nan_grads", False)

    if scenario.ground_truth.bug_type == "none":
        return 1.0, 0.0, "No bug; loss-only logger correctly sees no anomaly"

    detected = _has_nan_loss(losses)
    note_parts = []

    if not detected and len(losses) >= 3:
        baseline = sum(losses[:3]) / 3.0
        if baseline > 0:
            max_loss = max(losses)
            if max_loss > baseline * 5.0:
                detected = True
                note_parts.append(
                    f"Loss spike detected ({max_loss:.2f} vs baseline {baseline:.2f})"
                )

    if not detected and nan_grads:
        note_parts.append(
            f"Loss is finite at every step; NaN gradients on parameters "
            f"not visible to loss-only logger"
        )
    elif not detected:
        note_parts.append("Loss is finite at every step; no anomaly flagged")

    notes = "; ".join(note_parts) if note_parts else "Anomaly detected via loss"
    return 1.0 if detected else 0.0, 0.0, notes


def _score_neuraldbg(scenario):
    """Score NeuralDBG using the standard evaluate() path."""
    dbg, gt = run_scenario(scenario)
    scores = evaluate(dbg, gt)
    return scores["detection"], scores["localization"], scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_comparison():
    results = {}
    tool_totals = {
        "neuraldbg": {"detection": 0.0, "localization": 0.0},
        "wandb": {"detection": 0.0, "localization": 0.0},
        "mlflow": {"detection": 0.0, "localization": 0.0},
        "tensorboard": {"detection": 0.0, "localization": 0.0},
    }
    n = len(PUBLIC_SCENARIOS)

    for scenario in PUBLIC_SCENARIOS:
        print(f"  Running {scenario.name} ...")
        nd_d, nd_l, nd_detail = _score_neuraldbg(scenario)
        wb_result = _train_wandb(scenario)
        wb_d, wb_l, wb_notes = _score_loss_logger(wb_result, scenario)
        ml_result = _train_mlflow(scenario)
        ml_d, ml_l, ml_notes = _score_loss_logger(ml_result, scenario)
        tb_result = _train_tensorboard(scenario)
        tb_d, tb_l, tb_notes = _score_loss_logger(tb_result, scenario)

        results[scenario.name] = {
            "neuraldbg": {
                "detection": nd_d,
                "localization": nd_l,
                "notes": "Hooks detect gradient anomalies at parameter level",
            },
            "wandb": {"detection": wb_d, "localization": wb_l, "notes": wb_notes},
            "mlflow": {"detection": ml_d, "localization": ml_l, "notes": ml_notes},
            "tensorboard": {"detection": tb_d, "localization": tb_l, "notes": tb_notes},
        }
        for t, d, l in [
            ("neuraldbg", nd_d, nd_l),
            ("wandb", wb_d, wb_l),
            ("mlflow", ml_d, ml_l),
            ("tensorboard", tb_d, tb_l),
        ]:
            tool_totals[t]["detection"] += d
            tool_totals[t]["localization"] += l

    summary = {
        t: {
            "avg_detection": round(v["detection"] / n, 2),
            "avg_localization": round(v["localization"] / n, 2),
        }
        for t, v in tool_totals.items()
    }

    payload = {
        "benchmark": "neuralsuite-tool-comparison-v1",
        "version": "1.3.2",
        "scenarios": results,
        "summary": summary,
    }

    out_path = Path(__file__).resolve().parent / "comparison_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    _print_table(summary)
    return payload


def _print_table(summary):
    print("\nTool Comparison -- 4 scenarios")
    print("=" * 72)
    print(f"{'Tool':<16}| {'Detection':>10} | {'Localization':>13} | Notes")
    print("-" * 72)
    notes_map = {
        "neuraldbg": "Hooks detect NaN gradients at parameter level",
        "wandb": "Logs loss only; misses NaN gradients when loss is finite",
        "mlflow": "Logs loss only; no gradient health monitoring",
        "tensorboard": "Logs loss only; no gradient health monitoring",
    }
    label_map = {
        "neuraldbg": "NeuralDBG",
        "wandb": "W&B (offline)",
        "mlflow": "MLflow",
        "tensorboard": "TensorBoard",
    }
    for key in ["neuraldbg", "wandb", "mlflow", "tensorboard"]:
        s = summary[key]
        print(
            f"{label_map[key]:<16}| {s['avg_detection']:>10.2f} | "
            f"{s['avg_localization']:>13.2f} | {notes_map[key]}"
        )
    print("=" * 72)


if __name__ == "__main__":
    run_comparison()
