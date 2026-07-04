"""neuraldbg.cli — Command-line wrapper for zero-code injection.

Usage:
    neuraldbg run training.py              # Auto-inject hooks, run, print report
    neuraldbg run training.py --export aquarium.json  # Also export for Aquarium
    neuraldbg run training.py --agent      # Also run Neural-Agent auto-fix
    neuraldbg run training.py --dry-run    # Show injected code without running
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _find_training_script(path: str) -> Path:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Training script not found: {p}")
    if not p.suffix == ".py":
        raise ValueError(f"Expected a .py file, got: {p}")
    return p


def _export_events(dbg, export_path: Optional[Path]) -> Optional[str]:
    """Export captured events as a JSON file consumable by Aquarium."""
    if export_path is None:
        return None
    events = dbg.dump_events()  # list of dicts
    record = {
        "source": "neuraldbg",
        "version": getattr(dbg, "VERSION", "1.3.2"),
        "events": events,
        "summary": {
            "total_events": len(events),
            "event_types": list({e.get("event_type", "?") for e in events}),
            "gradient_health_transitions": [
                e for e in events
                if e.get("event_type") == "gradient_health_transition"
            ],
        },
    }
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return str(export_path)


def _print_report(dbg) -> None:
    """Print a human-readable summary of captured events."""
    events = dbg.dump_events()
    print("\n" + "=" * 60)
    print("  NeuralDBG Run Report")
    print("=" * 60)
    print(f"  Total events captured: {len(events)}")

    by_type: Dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "unknown")
        by_type[et] = by_type.get(et, 0) + 1

    for et, count in sorted(by_type.items()):
        print(f"    {et}: {count}")

    # Highlight critical issues (exclude healthy transitions)
    critical = [e for e in events if (
        e.get("event_type") in ("gradient_health_transition", "nan_detected", "silent_corruption")
        and (
            e.get("event_type") != "gradient_health_transition"
            or e.get("to_state", "") not in ("healthy", "none", "NONE", "")
        )
    )]
    if critical:
        print(f"\n  [WARN] {len(critical)} critical event(s) detected:")
        for e in critical:
            print(f"    - {e.get('event_type')}: {json.dumps(e, default=str)[:120]}")
    else:
        print(f"\n  [OK] No critical events detected.")
    print("=" * 60 + "\n")


def run_script(script_path: Path, export_path: Optional[Path], agent: bool, dry_run: bool):
    """Main runner: inject NeuralDBG hooks and execute the training script."""

    # Phase 1: Read source
    source = script_path.read_text(encoding="utf-8")

    # Phase 2: Inject NeuralDBG wrapper via AST rewriting
    from neuraldbg.injector import inject_neuraldbg_wrapper

    injected = inject_neuraldbg_wrapper(source, script_path.stem)
    if dry_run:
        print("=== Injected source (dry-run) ===")
        print(injected)
        return

    # Phase 3: Execute the injected code in-process
    injected_path = script_path.parent / f"_neuraldbg_{script_path.stem}.py"
    try:
        injected_path.write_text(injected, encoding="utf-8")

        # Run with the original script's __name__ and path
        original_argv = sys.argv[:]
        original_cwd = Path.cwd()
        try:
            sys.argv = [str(script_path)]  # mimic original invocation
            os_chdir = script_path.parent
            import os
            os.chdir(str(os_chdir))

            # Execute in a way that captures NeuralDbg instance
            runpy.run_path(str(injected_path), run_name="__main__")
        finally:
            sys.argv = original_argv
            os.chdir(str(original_cwd))
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"[NeuralDBG] Script exited with code {e.code}")
    except Exception as e:
        print(f"[NeuralDBG] Script raised {type(e).__name__}: {e}")
        raise
    finally:
        # Clean up injected file unless kept for debugging
        if not dry_run and "_KEEP_INJECTED" not in os.environ:
            injected_path.unlink(missing_ok=True)

    # Phase 4: Print report + optional agent
    events_path = injected_path.with_suffix(".events.json")
    if events_path.exists():
        _print_report_from_file(events_path, export_path, agent, script_path if agent else None)
        events_path.unlink(missing_ok=True)
    else:
        print("[NeuralDBG] No events captured (did training loop run?)")


def _print_report_from_file(
    events_path: Path, export_path: Optional[Path], agent: bool, script_path: Optional[Path]
):
    """Read events from JSON file, generate reports, optionally run agent."""
    with open(events_path, encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    print("\n" + "=" * 60)
    print("  NeuralDBG Run Report")
    print("=" * 60)
    print(f"  Total events captured: {len(events)}")

    by_type: Dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "unknown")
        by_type[et] = by_type.get(et, 0) + 1

    for et, count in sorted(by_type.items()):
        print(f"    {et}: {count}")

    critical = [e for e in events if (
        e.get("event_type") in ("gradient_health_transition", "nan_detected", "silent_corruption")
        and (
            e.get("event_type") != "gradient_health_transition"
            or e.get("to_state", "") not in ("healthy", "none", "NONE", "")
        )
    )]
    if critical:
        print(f"\n  [WARN] {len(critical)} critical event(s) detected:")
        for e in critical:
            print(f"    - {e.get('event_type')}: {json.dumps(e, default=str)[:120]}")
    else:
        print(f"\n  [OK] No critical events detected.")

    # Export for Aquarium
    if export_path:
        record = {
            "source": "neuraldbg",
            "version": "1.3.2",
            "events": events,
            "summary": {
                "total_events": len(events),
                "event_types": list(by_type.keys()),
                "critical_count": len(critical),
            },
        }
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        print(f"\n  Exported to: {export_path}")

    # Run Neural-Agent auto-fix
    if agent and critical:
        print(f"\n  [Agent] Running Neural-Agent auto-fix...")
        _run_neuralagent_fix(events, script_path)

    print("=" * 60 + "\n")


def _run_neuralagent_fix(events: list, script_path: Optional[Path]) -> None:
    """Run Neural-Agent remediation on detected critical events."""
    try:
        import neuralagent
    except ImportError:
        print("  [Agent] neuralagent not installed. Skipping auto-fix.")
        return

    # Phase 1: Try LLM-based fix if trained model is available
    print(f"  [Agent] Trying LLM-based diagnosis...")
    llm_suggestion = ""
    try:
        from neuralagent.llm_bridge import suggest_fix
        import json as _json
        combined = " ".join([_json.dumps(e, default=str) for e in events[:3]])
        llm_suggestion = suggest_fix(combined)
        if llm_suggestion:
            print(f"  [Agent] LLM: {llm_suggestion[:200]}")
        else:
            print(f"  [Agent] LLM model not available (no adapter found). Falling back to rules.")
    except Exception as e:
        print(f"  [Agent] LLM error: {e}. Falling back to rules.")

    # Phase 2: Rule-based classification (always runs, supplements LLM)
    from neuralagent.remediation_rules import classify_hypothesis, REMEDIATION_STRATEGIES

    # Classify each critical event and collect remediation strategies
    fixes: Dict[str, dict] = {}
    for e in events:
        desc = json.dumps(e, default=str)
        category = classify_hypothesis(desc)
        strategy = REMEDIATION_STRATEGIES.get(category, {})
        if strategy:
            fixes[category] = strategy
            print(f"    Category: {category}")
            print(f"      Fix: {strategy.get('description', 'N/A')}")
            for key, val in strategy.items():
                if key != "description":
                    print(f"      -> {key} = {val}")

    if not fixes:
        print("  [Agent] No remediation strategies matched.")
        return

    # Apply fixes via ScriptRewriter if we have a script
    if script_path and script_path.exists():
        from neuralagent.code.rewriter import ScriptRewriter
        original = script_path.read_text(encoding="utf-8")
        rewriter = ScriptRewriter()

        # Apply all remediation strategies as fix dicts
        modified = original
        for category, strategy in fixes.items():
            if category == "mha_fully_masked_row" and strategy.get("mha_mask_merge"):
                modified = rewriter.apply_fix(modified, {
                    "type": "code_change",
                    "action": "apply_mha_mask_fix",
                })
            elif "lr_multiplier" in strategy:
                modified = rewriter.apply_fix(modified, {
                    "type": "hyperparameter",
                    "lr_multiplier": strategy["lr_multiplier"],
                })
            if strategy.get("clip_grad_norm"):
                modified = rewriter.apply_fix(modified, {
                    "type": "hyperparameter",
                    "action": "clip_grad",
                    "clip_grad_norm": strategy["clip_grad_norm"],
                })

        if modified != original:
            backup = script_path.with_suffix(script_path.suffix + ".bak")
            script_path.rename(backup)
            script_path.write_text(modified, encoding="utf-8")
            print(f"\n  [Agent] Applied fixes to {script_path.name}")
            print(f"  [Agent] Original backed up as {backup.name}")
            print(f"  [Agent] Re-run: neuraldbg run {script_path.name}")
        else:
            print(f"  [Agent] No source changes needed.")
    else:
        print(f"\n  [Agent] No script path provided. Fixes above can be applied manually.")


def main(args: Optional[list] = None):
    parser = argparse.ArgumentParser(
        prog="neuraldbg",
        description="NeuralDBG — Causal inference for DL training. Zero-code injection via CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_parser = sub.add_parser("run", help="Run a training script with NeuralDBG auto-injected")
    run_parser.add_argument("script", help="Path to the training .py script")
    run_parser.add_argument("--export", type=Path, default=None, help="Export events for Aquarium (JSON)")
    run_parser.add_argument("--agent", action="store_true", help="Run Neural-Agent auto-fix after training")
    run_parser.add_argument("--dry-run", action="store_true", help="Print injected code without executing")

    ns = parser.parse_args(args)
    if ns.command == "run":
        script_path = _find_training_script(ns.script)
        run_script(script_path, ns.export, ns.agent, ns.dry_run)


if __name__ == "__main__":
    main()