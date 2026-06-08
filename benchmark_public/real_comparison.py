"""Honest tool comparison: NeuralDBG vs W&B vs MLflow vs TensorBoard.

Two modes per external tool:
  1. Loss only (default usage — what 90% of users do)
  2. Loss + gradient norms (power user — monitors grad health)

NeuralDBG is always in its native mode (causal hooks + composite detection).

Scoring criteria:
  - detection: 1.0 if the tool WOULD flag the bug given what it logged
  - localization: 1.0 if the tool can name the LAYER that caused the bug
  - For healthy_training: both detection and localization are N/A (excluded
    from averages to avoid inflating scores with trivial true negatives)

Output: benchmark_public/comparison_results.json + stdout table.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from benchmark_public.scenarios import PUBLIC_SCENARIOS
from benchmark_public.run import run_scenario, evaluate

SEED = 42


def _train_wandb(scenario, log_grads: bool = False):
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
        grad_norms = []
        nan_grad_steps = []
        has_nan = False
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        if scenario.step_fn is not None:
            for step in range(scenario.num_steps):
                scenario.bug_injector(model, step)
                x_t, am, kpm = data[0], data[1], data[2]
                out, _ = model(x_t, x_t, x_t, attn_mask=am, key_padding_mask=kpm)
                loss = out[:2, :].sum()
                loss.backward()
                lv = loss.item()
                losses.append(lv)
                log = {"loss": lv, "step": step}
                step_nan = False
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            step_nan = True
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm**0.5
                grad_norms.append(total_norm)
                if log_grads:
                    log["grad_norm_total"] = total_norm
                wandb.log(log)
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                model.zero_grad()
        else:
            x, y = data
            for step in range(scenario.num_steps):
                optimizer.zero_grad()
                scenario.bug_injector(model, step)
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                lv = loss.item()
                losses.append(lv)
                log = {"loss": lv, "step": step}
                step_nan = False
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            step_nan = True
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm**0.5
                grad_norms.append(total_norm)
                if log_grads:
                    log["grad_norm_total"] = total_norm
                wandb.log(log)
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                optimizer.step()

        wandb.finish()
        return {
            "losses": losses,
            "grad_norms": grad_norms,
            "has_nan_grads": has_nan,
            "nan_grad_steps": nan_grad_steps,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _train_mlflow(scenario, log_grads: bool = False):
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
            grad_norms = []
            nan_grad_steps = []
            has_nan = False
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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
                    step_nan = False
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any():
                                step_nan = True
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm**0.5
                    grad_norms.append(total_norm)
                    if log_grads:
                        mlflow.log_metric("grad_norm_total", total_norm, step=step)
                    if step_nan:
                        nan_grad_steps.append(step)
                        has_nan = True
                    model.zero_grad()
            else:
                x, y = data
                for step in range(scenario.num_steps):
                    optimizer.zero_grad()
                    scenario.bug_injector(model, step)
                    out = model(x)
                    loss = nn.CrossEntropyLoss()(out, y)
                    loss.backward()
                    lv = loss.item()
                    losses.append(lv)
                    mlflow.log_metric("loss", lv, step=step)
                    step_nan = False
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any():
                                step_nan = True
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm**0.5
                    grad_norms.append(total_norm)
                    if log_grads:
                        mlflow.log_metric("grad_norm_total", total_norm, step=step)
                    if step_nan:
                        nan_grad_steps.append(step)
                        has_nan = True
                    optimizer.step()

        return {
            "losses": losses,
            "grad_norms": grad_norms,
            "has_nan_grads": has_nan,
            "nan_grad_steps": nan_grad_steps,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _train_tensorboard(scenario, log_grads: bool = False):
    from torch.utils.tensorboard import SummaryWriter

    tmpdir = tempfile.mkdtemp(prefix="bench_tb_")
    try:
        writer = SummaryWriter(log_dir=tmpdir)
        torch.manual_seed(SEED)
        model = scenario.model_builder()
        data = scenario.data_builder()
        losses = []
        grad_norms = []
        nan_grad_steps = []
        has_nan = False
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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
                step_nan = False
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            step_nan = True
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm**0.5
                grad_norms.append(total_norm)
                if log_grads:
                    writer.add_scalar("grad_norm_total", total_norm, step)
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                model.zero_grad()
        else:
            x, y = data
            for step in range(scenario.num_steps):
                optimizer.zero_grad()
                scenario.bug_injector(model, step)
                out = model(x)
                loss = nn.CrossEntropyLoss()(out, y)
                loss.backward()
                lv = loss.item()
                losses.append(lv)
                writer.add_scalar("loss", lv, step)
                step_nan = False
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any():
                            step_nan = True
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm**0.5
                grad_norms.append(total_norm)
                if log_grads:
                    writer.add_scalar("grad_norm_total", total_norm, step)
                if step_nan:
                    nan_grad_steps.append(step)
                    has_nan = True
                optimizer.step()

        writer.close()
        return {
            "losses": losses,
            "grad_norms": grad_norms,
            "has_nan_grads": has_nan,
            "nan_grad_steps": nan_grad_steps,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _has_nan_loss(losses):
    return any(
        isinstance(l, float) and (math.isnan(l) or math.isinf(l)) for l in losses
    )


def _score_tool(result, scenario, log_grads: bool):
    """Score any external tool based on what it logged.

    loss_only mode:
      - detection: NaN loss OR loss spike > 5x baseline
      - localization: always 0.0

    grad_norm mode:
      - detection: NaN loss OR loss spike > 5x OR NaN gradient norm
      - localization: 0.0 (tool sees NaN but can't name the layer)
    """
    losses = result["losses"]
    grad_norms = result.get("grad_norms", [])
    nan_grads = result.get("has_nan_grads", False)
    nan_steps = result.get("nan_grad_steps", [])

    if scenario.ground_truth.bug_type == "none":
        return None, None, "No bug (excluded from averages)"

    detected = _has_nan_loss(losses)
    note_parts = []

    if not detected and len(losses) >= 3:
        baseline = sum(losses[:3]) / 3.0
        if baseline > 0:
            max_loss = max(losses)
            if max_loss > baseline * 5.0:
                detected = True
                note_parts.append(
                    f"Loss spike: {max_loss:.2f} vs baseline {baseline:.2f}"
                )

    if not detected and log_grads and nan_grads:
        nan_norm_steps = [
            i for i, n in enumerate(grad_norms) if math.isnan(n) or math.isinf(n)
        ]
        if nan_norm_steps:
            detected = True
            note_parts.append(f"NaN grad_norm at step(s) {nan_norm_steps}")
        elif nan_grads:
            detected = True
            note_parts.append(f"NaN gradients detected via grad_norm monitoring")

    if not detected and nan_grads:
        note_parts.append(f"NaN grads present but NOT logged (loss-only mode)")
    elif not detected:
        note_parts.append("No anomaly visible in logged data")

    notes = "; ".join(note_parts) if note_parts else "Anomaly detected"
    return (1.0 if detected else 0.0), 0.0, notes


def run_comparison():
    results = {}
    n_buggy = 0

    for scenario in PUBLIC_SCENARIOS:
        print(f"  Running {scenario.name} ...")

        nd_d, nd_l, nd_detail = _score_neuraldbg(scenario)

        wb_loss = _train_wandb(scenario, log_grads=False)
        wb_d_loss, _, wb_notes_loss = _score_tool(wb_loss, scenario, log_grads=False)

        wb_grad = _train_wandb(scenario, log_grads=True)
        wb_d_grad, _, wb_notes_grad = _score_tool(wb_grad, scenario, log_grads=True)

        ml_loss = _train_mlflow(scenario, log_grads=False)
        ml_d_loss, _, ml_notes_loss = _score_tool(ml_loss, scenario, log_grads=False)

        ml_grad = _train_mlflow(scenario, log_grads=True)
        ml_d_grad, _, ml_notes_grad = _score_tool(ml_grad, scenario, log_grads=True)

        tb_loss = _train_tensorboard(scenario, log_grads=False)
        tb_d_loss, _, tb_notes_loss = _score_tool(tb_loss, scenario, log_grads=False)

        tb_grad = _train_tensorboard(scenario, log_grads=True)
        tb_d_grad, _, tb_notes_grad = _score_tool(tb_grad, scenario, log_grads=True)

        is_healthy = scenario.ground_truth.bug_type == "none"
        if not is_healthy:
            n_buggy += 1

        results[scenario.name] = {
            "neuraldbg": {
                "detection": nd_d,
                "localization": nd_l,
                "notes": "Causal hooks + composite detection",
            },
            "wandb_loss_only": {
                "detection": wb_d_loss,
                "localization": 0.0,
                "notes": wb_notes_loss,
            },
            "wandb_grad_norms": {
                "detection": wb_d_grad,
                "localization": 0.0,
                "notes": wb_notes_grad,
            },
            "mlflow_loss_only": {
                "detection": ml_d_loss,
                "localization": 0.0,
                "notes": ml_notes_loss,
            },
            "mlflow_grad_norms": {
                "detection": ml_d_grad,
                "localization": 0.0,
                "notes": ml_notes_grad,
            },
            "tensorboard_loss_only": {
                "detection": tb_d_loss,
                "localization": 0.0,
                "notes": tb_notes_loss,
            },
            "tensorboard_grad_norms": {
                "detection": tb_d_grad,
                "localization": 0.0,
                "notes": tb_notes_grad,
            },
        }

    tools = [
        "neuraldbg",
        "wandb_loss_only",
        "wandb_grad_norms",
        "mlflow_loss_only",
        "mlflow_grad_norms",
        "tensorboard_loss_only",
        "tensorboard_grad_norms",
    ]
    summary = {}
    for tool in tools:
        d_sum = 0.0
        l_sum = 0.0
        for scenario_name, sc_results in results.items():
            is_healthy = (
                PUBLIC_SCENARIOS[
                    [s.name for s in PUBLIC_SCENARIOS].index(scenario_name)
                ].ground_truth.bug_type
                == "none"
            )
            if not is_healthy:
                d_sum += sc_results[tool]["detection"]
                l_sum += sc_results[tool]["localization"]
        n = max(n_buggy, 1)
        summary[tool] = {
            "avg_detection": round(d_sum / n, 2),
            "avg_localization": round(l_sum / n, 2),
        }

    payload = {
        "benchmark": "neuralsuite-tool-comparison-v2",
        "version": "1.3.2",
        "note": "Healthy scenarios excluded from averages (trivial true "
        "negatives). Loss-only = default user. Grad-norms = power "
        "user who monitors gradient health.",
        "scenarios": results,
        "summary": summary,
        "n_buggy_scenarios": n_buggy,
    }

    out_path = Path(__file__).resolve().parent / "comparison_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    _print_table(summary)
    return payload


def _print_table(summary):
    print("\nTool Comparison -- 3 buggy scenarios (healthy excluded)")
    print("=" * 95)
    print(f"{'Tool':<28} | {'Det(loss)':>9} | {'Det(+grad)':>10} | {'Local':>5}")
    print("-" * 95)

    def _d(key):
        return summary.get(key, {}).get("avg_detection", 0.0)

    def _l(key):
        return summary.get(key, {}).get("avg_localization", 0.0)

    merged = [
        ("NeuralDBG (causal)", _d("neuraldbg"), _d("neuraldbg"), _l("neuraldbg")),
        ("W&B", _d("wandb_loss_only"), _d("wandb_grad_norms"), _l("wandb_loss_only")),
        (
            "MLflow",
            _d("mlflow_loss_only"),
            _d("mlflow_grad_norms"),
            _l("mlflow_loss_only"),
        ),
        (
            "TensorBoard",
            _d("tensorboard_loss_only"),
            _d("tensorboard_grad_norms"),
            _l("tensorboard_loss_only"),
        ),
    ]
    for label, d_loss, d_grad, loc in merged:
        print(f"{label:<28} | {d_loss:>9.2f} | {d_grad:>10.2f} | {loc:>5.2f}")
    print("=" * 95)
    print("Det(loss) = detection using loss metrics only (default usage)")
    print("Det(+grad) = detection adding gradient norm monitoring")
    print("Local = can the tool NAME the failing layer?")
    print("Healthy scenarios excluded from averages.")


def _score_neuraldbg(scenario):
    dbg, gt = run_scenario(scenario)
    scores = evaluate(dbg, gt)
    return scores["detection"], scores["localization"], scores


if __name__ == "__main__":
    run_comparison()
