"""
NeuralSuite — Full R108 Pipeline Runner + Self-Evolution Loop.

Runs the complete validation pipeline and auto-retrains Neural-Agent
until all metrics converge to their best values.

Pipeline stages (R108):
  Stage 1: Fuzzer       — discover new failure modes
  Stage 2: Stress       — resilience verification (15/15)
  Stage 3: Combinatorial — coverage across 6 families
  Stage 4: OOS          — anti-overfitting gate (4 architectures)
  Stage 5: Benchmark    — superiority verification

After each run:
  - Gaps are identified (weakest family, lowest bug type, highest FP)
  - Training data is generated from gaps
  - Neural-Agent is retrained on expanded dataset
  - Pipeline re-runs
  - Repeats until all metrics plateau (3 runs without improvement)

Usage:
  python run_pipeline.py              # Quick mode (50 configs, no retrain)
  python run_pipeline.py --full       # Full mode (200 configs, auto-retrain)
  python run_pipeline.py --loop       # Loop until convergence
  python run_pipeline.py --stage oos  # Run single stage
"""

import sys, os, json, time, subprocess, argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Configuration
# ============================================================
BASE = Path(__file__).parent
AGENT_DIR = Path(r"C:\Users\Utilisateur\Documents\Neural-Agent")
VENV_PYTHON = BASE / ".venv" / "Scripts" / "python.exe"
CONVERGENCE_PATIENCE = 3  # Stop after N runs without improvement

STAGE_SCRIPTS = {
    "fuzzer":        ("arch_fuzzer.py",         ["--runs", "20"]),
    "stress":        ("stress_test_suite.py",    []),
    "combinatorial": ("validate_combinatorial.py", ["--quick"]),
    "oos":           ("validate_oos.py",          []),
    "benchmark":     ("benchmark_honest.py",      []),
}

STAGE_GATES = {
    "fuzzer":        {"metric": "crash_rate",     "min": 0},    # Any crashes found = bug discovery success
    "stress":        {"metric": "pass_rate",      "min": 93},   # 14/15 minimum
    "combinatorial": {"metric": "detection_pct",  "min": 85},   # Raised from 80
    "oos":           {"metric": "detection_rate", "min": 100},
    "benchmark":     {"metric": "advantage",      "min": 0},
}


# ============================================================
# Stage runners
# ============================================================

def run_stage(stage_name: str, extra_args: list = None) -> dict:
    """Run a single pipeline stage and parse its results."""
    if stage_name not in STAGE_SCRIPTS:
        return {"error": f"Unknown stage: {stage_name}", "success": False}

    script, default_args = STAGE_SCRIPTS[stage_name]
    args = [str(VENV_PYTHON), str(BASE / script)] + default_args
    if extra_args:
        args.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name.upper()}")
    print(f"  Command: {' '.join(args[-3:])}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=600,
            cwd=str(BASE),
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s (exit={result.returncode})")
        if result.stdout:
            # Print last 5 lines of output
            lines = result.stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"  | {line[:120]}")
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (10min)", "success": False, "elapsed": 600}
    except Exception as e:
        return {"error": str(e)[:100], "success": False}

    return {
        "success": result.returncode == 0,
        "elapsed": elapsed,
        "stdout_tail": result.stdout.strip().split("\n")[-10:] if result.stdout else [],
    }


def parse_results(stage_name: str) -> dict:
    """Parse the output files for a stage into structured metrics."""
    metrics = {"stage": stage_name}

    if stage_name == "fuzzer":
        try:
            with open(BASE / "fuzz_report.json") as f:
                d = json.load(f)
            crashes = d.get("crashes", 0)
            total = d.get("total", 20)
            metrics["crash_rate"] = round(100 * crashes / max(total, 1))
            metrics["crashes"] = crashes
            metrics["total"] = total
        except Exception:
            metrics["crash_rate"] = 0

    elif stage_name == "stress":
        try:
            output_file = BASE / "stress_output.txt"
            if output_file.exists():
                content = output_file.read_text()
                if "ALL RESILIENCE TESTS PASSED" in content:
                    metrics["pass_rate"] = 100
                    metrics["passed"] = 15
                else:
                    # Count PASS lines
                    import re
                    passes = len(re.findall(r'PASS', content))
                    metrics["pass_rate"] = round(100 * passes / 15)
                    metrics["passed"] = passes
            else:
                metrics["pass_rate"] = 0
        except Exception:
            metrics["pass_rate"] = 0

    elif stage_name == "combinatorial":
        try:
            with open(BASE / "combinatorial_results.json") as f:
                d = json.load(f)
            overall = d["overall_detection"]
            detected, total = overall.split("/")
            metrics["detection_pct"] = round(100 * int(detected) / int(total))
            metrics["detected"] = int(detected)
            metrics["total"] = int(total)
            metrics["by_family"] = d.get("by_family", {})
            metrics["by_bug"] = d.get("by_bug", {})
        except Exception:
            metrics["detection_pct"] = 0

    elif stage_name == "oos":
        try:
            with open(BASE / "oos_validation_report.json") as f:
                d = json.load(f)
            rate = d.get("detection_rate", "0/0")
            detected, total = rate.split("/")
            metrics["detection_rate"] = round(100 * int(detected) / max(int(total), 1))
            metrics["detected"] = int(detected)
            metrics["total"] = int(total)
            metrics["architectures"] = d.get("architectures", [])
        except Exception:
            metrics["detection_rate"] = 0

    elif stage_name == "benchmark":
        try:
            with open(BASE / "benchmark_honest.json") as f:
                d = json.load(f)
            nd = d.get("neuraldbg_detection", "0/0")
            wb = d.get("wandb_detection", "0/0")
            nd_d, nd_t = nd.split("/")
            wb_d, wb_t = wb.split("/")
            metrics["neuraldbg_detection"] = round(100 * int(nd_d) / max(int(nd_t), 1))
            metrics["wandb_detection"] = round(100 * int(wb_d) / max(int(wb_t), 1))
            metrics["advantage"] = int(nd_d) - int(wb_d)
            metrics["causal_chains"] = d.get("neuraldbg_causal_chains", 0)
        except Exception:
            metrics["advantage"] = 0

    return metrics


# ============================================================
# Gate checker
# ============================================================

def check_gate(stage_name: str, metrics: dict) -> tuple:
    """Check if a stage passed its gate. Returns (passed: bool, detail: str)."""
    gate = STAGE_GATES.get(stage_name)
    if not gate:
        return True, "no gate defined"

    metric_key = gate["metric"]
    value = metrics.get(metric_key, 0)
    passed = value >= gate["min"]

    detail = f"{metric_key}={value} (gate: >={gate['min']})"
    return passed, detail


# ============================================================
# Gap analysis
# ============================================================

def find_gaps(all_metrics: dict) -> list:
    """Identify the weakest areas across all stages."""
    gaps = []

    # Combinatorial: weakest family
    comb = all_metrics.get("combinatorial", {})
    by_family = comb.get("by_family", {})
    for fam, v in by_family.items():
        fam_det, fam_tot = v.get("detection", "0/0").split("/")
        fam_pct = round(100 * int(fam_det) / max(int(fam_tot), 1))
        if fam_pct < 70:
            gaps.append({
                "type": "weak_family",
                "family": fam,
                "detection": fam_pct,
                "action": f"Generate more training data for {fam} architectures",
            })

    # Combinatorial: weakest bug type
    by_bug = comb.get("by_bug", {})
    for bug, v in by_bug.items():
        bug_det = v.get("detected", 0)
        bug_tot = v.get("total", 1)
        bug_pct = round(100 * bug_det / max(bug_tot, 1))
        if bug_pct < 70:
            gaps.append({
                "type": "weak_bug",
                "bug": bug,
                "detection": bug_pct,
                "action": f"Improve {bug} detection logic in neuraldbg hooks",
            })

    # OOS: failed architectures
    oos = all_metrics.get("oos", {})
    if oos.get("detection_rate", 100) < 100:
        gaps.append({
            "type": "oos_failure",
            "detection": oos.get("detection_rate", 0),
            "action": "OOS gate failed — NeuralDBG is overfitted. STOP and fix.",
        })

    # Benchmark: if not superior
    bench = all_metrics.get("benchmark", {})
    if bench.get("advantage", 0) <= 0:
        gaps.append({
            "type": "benchmark_gap",
            "advantage": bench.get("advantage", 0),
            "action": "NeuralDBG not superior to baseline — improve detection",
        })

    return gaps


# ============================================================
# Training data generation + Neural-Agent retraining
# ============================================================

def generate_training_data(gaps: list) -> int:
    """Generate new training triplets from detected gaps."""
    print(f"\n  Generating training data from {len(gaps)} gaps...")

    # Collect event data from last combinatorial run
    try:
        with open(BASE / "combinatorial_results.json") as f:
            d = json.load(f)
        results = d.get("results", [])
    except Exception:
        return 0

    new_triplets = []
    for r in results:
        for bug_result in r.get("results", []):
            if not bug_result.get("detected", True):
                # This is a missed detection — valuable training data
                new_triplets.append({
                    "instruction": (
                        f"Le modèle {r.get('family', 'MLP')} {r.get('name', '')} "
                        f"a un bug de type {bug_result.get('bug', '')}. "
                        f"NeuralDBG n'a pas détecté cette anomalie. "
                        f"Analyse ce qui aurait dû être détecté et propose un diagnostic."
                    ),
                    "response": {
                        "failure_type": bug_result.get("bug", "unknown"),
                        "root_cause": f"Non détecté dans {r.get('family', 'MLP')}",
                        "fix": {"lr_multiplier": 1.0},
                        "severity": "high",
                    },
                })

    if new_triplets:
        output_file = BASE / "pipeline_training_data.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for t in new_triplets:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"  Generated {len(new_triplets)} new training triplets → {output_file}")
        return len(new_triplets)

    print("  No new training data generated (all gaps detected)")
    return 0


def retrain_neural_agent() -> bool:
    """Retrain Neural-Agent on expanded dataset."""
    train_script = AGENT_DIR / "train_cpu.py"
    if not train_script.exists():
        print("  ⚠ Neural-Agent train script not found — skipping retrain")
        return False

    print(f"\n  Retraining Neural-Agent...")
    result = subprocess.run(
        [str(VENV_PYTHON), str(train_script), "--steps", "100"],
        capture_output=True, text=True, timeout=600,
        cwd=str(AGENT_DIR),
    )
    success = result.returncode == 0
    if success:
        print(f"  ✅ Neural-Agent retrained successfully")
    else:
        print(f"  ❌ Retrain failed: {result.stderr[-200:] if result.stderr else 'unknown'}")
    return success


# ============================================================
# Full pipeline
# ============================================================

def run_full_pipeline(stages: list = None, max_loops: int = 5) -> dict:
    """Run the complete R108 pipeline, optionally in a loop."""
    if stages is None:
        stages = ["fuzzer", "stress", "combinatorial", "oos", "benchmark"]

    all_history = []
    best_metrics = {}
    runs_without_improvement = 0
    loop_count = 0

    while loop_count < max_loops:
        loop_count += 1
        print(f"\n{'#'*60}")
        print(f"# PIPELINE RUN {loop_count}/{max_loops}")
        print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        run_metrics = {}
        all_passed = True

        for stage_name in stages:
            # Run stage
            run_stage(stage_name)

            # Parse results
            metrics = parse_results(stage_name)
            run_metrics[stage_name] = metrics

            # Check gate
            passed, detail = check_gate(stage_name, metrics)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  Gate {stage_name}: {status} — {detail}")

            if not passed:
                all_passed = False
                if stage_name == "oos":
                    print(f"\n  🛑 OOS GATE FAILED — Pipeline stopped.")
                    print(f"  NeuralDBG is overfitted. Fix overfitting before continuing.")
                    return {"status": "oos_failed", "metrics": run_metrics, "history": all_history}

        all_history.append(run_metrics)

        # Check improvement
        improved = False
        for stage_name in stages:
            gate = STAGE_GATES.get(stage_name, {})
            key = gate.get("metric", "")
            current = run_metrics.get(stage_name, {}).get(key, 0)
            previous = best_metrics.get(stage_name, {}).get(key, 0)
            if current > previous:
                best_metrics[stage_name] = run_metrics[stage_name]
                improved = True

        if improved:
            runs_without_improvement = 0
            print(f"\n  📈 Improvement detected! Continuing...")
        else:
            runs_without_improvement += 1
            print(f"\n  ⏸ No improvement ({runs_without_improvement}/{CONVERGENCE_PATIENCE})")

        # Find gaps and generate training data
        gaps = find_gaps(run_metrics)
        if gaps and loop_count < max_loops:
            print(f"\n  🔍 Found {len(gaps)} gaps:")
            for g in gaps:
                print(f"    - [{g['type']}] {g.get('action', '')[:100]}")

            n_new = generate_training_data(gaps)
            if n_new > 0:
                retrain_neural_agent()

        # Convergence check
        if runs_without_improvement >= CONVERGENCE_PATIENCE:
            print(f"\n  🎯 CONVERGED — No improvement after {CONVERGENCE_PATIENCE} runs")
            break

        if all_passed:
            print(f"\n  ✅ ALL GATES PASSED — Pipeline complete!")
            break

    # Final report
    report = {
        "date": datetime.now().isoformat(),
        "runs": loop_count,
        "converged": runs_without_improvement >= CONVERGENCE_PATIENCE,
        "all_passed": all_passed,
        "best_metrics": best_metrics,
        "history": all_history,
    }

    report_file = BASE / "pipeline_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — {loop_count} runs")
    print(f"  Report: {report_file}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n  Final Metrics:")
    print(f"  {'Stage':<20} {'Metric':<20} {'Value':<10} {'Gate':<10}")
    print(f"  {'-'*20} {'-'*20} {'-'*10} {'-'*10}")
    for stage_name in stages:
        gate = STAGE_GATES.get(stage_name, {})
        key = gate.get("metric", "")
        value = best_metrics.get(stage_name, {}).get(key, "N/A")
        gate_val = gate.get("min", "N/A")
        print(f"  {stage_name:<20} {key:<20} {str(value):<10} >={gate_val}")

    return report


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralSuite R108 Pipeline Runner")
    parser.add_argument("--full", action="store_true", help="Full mode (200 configs)")
    parser.add_argument("--loop", action="store_true", help="Loop until convergence")
    parser.add_argument("--stage", type=str, help="Run single stage")
    parser.add_argument("--max-loops", type=int, default=5, help="Max pipeline loops")
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
        metrics = parse_results(args.stage)
        passed, detail = check_gate(args.stage, metrics)
        print(f"\n  {args.stage}: {metrics}")
        print(f"  Gate: {'PASS' if passed else 'FAIL'} -- {detail}")
    elif args.loop:
        print("🔄 Pipeline Loop Mode — running until convergence...")
        run_full_pipeline(max_loops=args.max_loops)
    else:
        print(">>> Single Pipeline Run")
        stages = ["fuzzer", "stress", "combinatorial", "oos", "benchmark"]
        run_full_pipeline(stages=stages, max_loops=1)
