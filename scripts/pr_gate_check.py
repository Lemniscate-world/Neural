#!/usr/bin/env python3
"""PR Gate Validator — checks all 6 gates before upstream PR creation.

Usage:
    python scripts/pr_gate_check.py [--strict]

Exit code 0 = all gates pass. Exit code 1 = one or more gates fail.

MANDATORY per DEV_RULES.md rule D4 and .github/PR_GATE.md.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def gate1_fix_not_warning() -> bool:
    """G1: Ensure no warnings.warn() in changed Python files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    changed = [f for f in result.stdout.splitlines() if f.endswith(".py")]
    if not changed:
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "--name-only"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        changed = [f for f in result.stdout.splitlines() if f.endswith(".py")]

    for fpath in changed:
        full = REPO_ROOT / fpath
        if full.exists():
            content = full.read_text(encoding="utf-8", errors="ignore")
            if "warnings.warn(" in content:
                print(f"  ❌ G1 FAIL: {fpath} contains warnings.warn()")
                print("     Replace with a behavior fix, not a warning.")
                return False

    print("  ✅ G1 PASS: No warnings.warn() in changed files")
    return True


def gate2_pr_exists() -> bool:
    """G2: Manual check — user must confirm PR is open."""
    print("  ⚠️  G2 MANUAL: Is this PR actually open on the upstream repo?")
    response = input("     URL of the upstream PR (or 'skip' if not yet): ").strip()
    if not response or response.lower() == "skip":
        print("     ❌ G2 FAIL: PR not yet created on upstream.")
        return False
    if "github.com" not in response and "gitlab" not in response:
        print("     ⚠️  Doesn't look like a PR URL, but accepted.")
    print(f"  ✅ G2 PASS: PR at {response}")
    return True


def gate3_hardware_independent() -> bool:
    """G3: Check that repro scripts don't hardcode unavailable hardware."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    changed = result.stdout.splitlines()
    if not changed:
        result = subprocess.run(
            ["git", "diff", "HEAD~1", "--name-only"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        changed = result.stdout.splitlines()

    hardware_only = [".cuda()", '.to("mps")', '.to("cuda")', "device='cuda'"]
    for fpath in changed:
        if not fpath.endswith(".py"):
            continue
        full = REPO_ROOT / fpath
        if full.exists():
            content = full.read_text(encoding="utf-8", errors="ignore")
            has_hw = any(hw in content for hw in hardware_only)
            has_cpu_fallback = ".to('cpu')" in content or ".to(device)" in content
            if has_hw and not has_cpu_fallback:
                print(f"  ⚠️  G3 WARNING: {fpath} uses GPU/MPS without CPU fallback")
                print("     Add a CPU path or Kaggle/Colab notebook reference.")

    print("  ✅ G3 PASS: Hardware check complete (warnings above are non-blocking)")
    return True


def gate4_neuraldbg_cited() -> bool:
    """G4: Ensure NeuralDBG is mentioned in PR-related files."""
    # Check common locations for PR description
    to_check = [
        REPO_ROOT / ".github" / "PR_TEMPLATES" / "upstream-fix.md",
    ]
    # Also check git diff for "neuraldbg" mention
    result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    diff = result.stdout
    if not diff:
        result = subprocess.run(
            ["git", "diff", "HEAD~1"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        diff = result.stdout

    if "neuraldbg" in diff.lower() or "NeuralDBG" in diff:
        print("  ✅ G4 PASS: NeuralDBG cited in changes")
        return True

    print("  ⚠️  G4 WARNING: NeuralDBG not found in diff.")
    print("     Make sure the upstream PR description cites NeuralDBG.")
    response = input("     Did you cite NeuralDBG in the PR? (y/n): ").strip().lower()
    return response == "y"


def gate5_followup_plan() -> bool:
    """G5: Check for follow-up plan."""
    print("  ⚠️  G5 MANUAL: Define your follow-up plan.")
    print("     Day 3 action (if no review):")
    d3 = input("       > ").strip()
    print("     Day 7 action (if no review):")
    d7 = input("       > ").strip()
    if d3 or d7:
        print(f"  ✅ G5 PASS: Plan defined (D3: {d3 or 'none'}, D7: {d7 or 'none'})")
        return True
    print("  ❌ G5 FAIL: No follow-up plan defined.")
    return False


def gate6_no_workaround() -> bool:
    """G6: Zero tolerance for the word 'workaround'."""
    result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    diff = result.stdout
    if not diff:
        result = subprocess.run(
            ["git", "diff", "HEAD~1"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        diff = result.stdout

    # Count occurrences (case-insensitive) in added lines only
    added_lines = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    workaround_lines = [l for l in added_lines if "workaround" in l.lower()]
    if workaround_lines:
        print(f"  ❌ G6 FAIL: {len(workaround_lines)} occurrence(s) of 'workaround':")
        for line in workaround_lines[:5]:
            print(f"     {line[:120]}")
        print("     Replace with 'fix', 'resolution', or 'mitigation'.")
        return False

    print("  ✅ G6 PASS: Zero occurrences of 'workaround'")
    return True


def main():
    print("=" * 60)
    print("PR GATE VALIDATOR — .github/PR_GATE.md")
    print("=" * 60)
    print()

    strict = "--strict" in sys.argv

    gates = [
        ("G1: FIX not warning", gate1_fix_not_warning),
        ("G2: PR exists upstream", gate2_pr_exists),
        ("G3: Hardware-independent repro", gate3_hardware_independent),
        ("G4: NeuralDBG cited", gate4_neuraldbg_cited),
        ("G5: Follow-up plan", gate5_followup_plan),
        ("G6: No workaround word", gate6_no_workaround),
    ]

    passed = 0
    failed = 0
    for name, gate_fn in gates:
        print(f"\n--- {name} ---")
        try:
            if gate_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(gates)} gates")
    print("=" * 60)

    if failed > 0:
        print("\n❌ GATE CHECK FAILED. Fix the issues above before creating the PR.")
        if not strict:
            print("   (Use --strict to block commit.)")
        sys.exit(1)
    else:
        print("\n✅ ALL GATES PASSED. You may create the PR. 🚀")
        print("   Remember: post on X/Discord/Reddit immediately after submitting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
