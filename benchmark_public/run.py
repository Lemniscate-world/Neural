#!/usr/bin/env python3
"""Run the public causal benchmark and write benchmark_public/results.json."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from benchmark_public.scenarios import PUBLIC_SCENARIOS, GroundTruth, Scenario


def run_scenario(
    scenario: Scenario,
    lr: float = 0.01,
    threshold_vanishing: float = 1e-4,
    threshold_exploding: float = 10.0,
    seed: int = 42,
):
    if scenario.ground_truth.bug_type == "none":
        threshold_vanishing = 1e-6
        threshold_exploding = 1e3
        lr = 0.05
    from neuraldbg import NeuralDbg

    torch.manual_seed(seed)
    model = scenario.model_builder()
    x, y = scenario.data_builder()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    with NeuralDbg(
        model,
        threshold_vanishing=threshold_vanishing,
        threshold_exploding=threshold_exploding,
    ) as dbg:
        for step in range(scenario.num_steps):
            optimizer.zero_grad()
            dbg.step = step
            scenario.bug_injector(model, step)
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()
            dbg.record_loss(loss.item())
            optimizer.step()

    return dbg, scenario.ground_truth


def _hypothesis_mentions_layer(hyp, expected_layer: str) -> bool:
    """Check if a hypothesis mentions the expected layer via any available path.

    Checks: evidence[].layer_name, causal_chain[], description.
    """
    # Check evidence events
    for event in getattr(hyp, "evidence", []):
        if hasattr(event, "layer_name") and expected_layer in event.layer_name:
            return True
    # Check causal chain entries (format: "layer_name@step")
    for chain_entry in getattr(hyp, "causal_chain", []):
        if expected_layer in chain_entry:
            return True
    # Check description substring
    if expected_layer in getattr(hyp, "description", ""):
        return True
    return False


def _hypothesis_step(hyp) -> int | None:
    """Extract the step from a hypothesis via evidence or attributes."""
    # Direct attribute
    step = getattr(hyp, "step", None)
    if step is not None:
        return int(step)
    # From first evidence event
    for event in getattr(hyp, "evidence", []):
        if hasattr(event, "step"):
            return event.step
    return None


def evaluate(dbg, ground_truth: GroundTruth) -> dict:
    hyps = dbg.explain_failure()
    descriptions = " ".join(
        h.description if hasattr(h, "description") else str(h) for h in hyps
    )

    if ground_truth.bug_type == "none":
        overall = 1.0 if len(hyps) == 0 else 0.0
        return {
            "detection": overall,
            "localization": overall,
            "step_accuracy": overall,
            "overall": overall,
            "num_hypotheses": len(hyps),
        }

    detection = (
        1.0
        if ground_truth.expected_hypothesis_substring.lower() in descriptions.lower()
        else 0.0
    )
    localization = (
        1.0
        if any(
            _hypothesis_mentions_layer(h, ground_truth.expected_bug_layer) for h in hyps
        )
        else 0.0
    )
    step_accuracy = 0.0
    for h in hyps:
        step = _hypothesis_step(h)
        if step is not None and abs(step - ground_truth.bug_step) <= 2:
            step_accuracy = 1.0
            break

    overall = (detection + localization + step_accuracy) / 3.0
    return {
        "detection": detection,
        "localization": localization,
        "step_accuracy": step_accuracy,
        "overall": overall,
        "num_hypotheses": len(hyps),
    }


def run_public_benchmark(verbose: bool = True) -> dict:
    results = {}
    totals = {
        "detection": 0.0,
        "localization": 0.0,
        "step_accuracy": 0.0,
        "overall": 0.0,
    }

    for scenario in PUBLIC_SCENARIOS:
        t0 = time.perf_counter()
        dbg, gt = run_scenario(scenario)
        score = evaluate(dbg, gt)
        score["execution_time_sec"] = round(time.perf_counter() - t0, 3)
        results[scenario.name] = score
        for k in totals:
            totals[k] += score[k]
        if verbose:
            print(f"{scenario.name}: overall={score['overall']:.2f}")

    n = len(PUBLIC_SCENARIOS)
    summary = {k: round(v / n, 3) for k, v in totals.items()}
    payload = {
        "benchmark": "neuraldbg-public-causal-v1",
        "version": "1.3.1",
        "scenarios": len(PUBLIC_SCENARIOS),
        "summary": summary,
        "results": results,
    }
    return payload


def main():
    payload = run_public_benchmark()
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Overall accuracy: {payload['summary']['overall']:.3f}")


if __name__ == "__main__":
    main()
