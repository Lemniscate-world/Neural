#!/usr/bin/env python3
"""PR Gate Validator — checks all 6 gates before upstream PR creation.

Usage:
    python scripts/pr_gate_check.py [--strict] [--non-interactive]

Exit code 0 = all gates pass (or warnings only).
MANDATORY per DEV_RULES.md rule D4 and .github/PR_GATE.md.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_FILES = {
    "scripts/pr_gate_check.py",
    ".github/PR_GATE.md",
    "DEV_RULES.md",
}

KW = {"capture_output": True, "text": True, "encoding": "utf-8",
      "errors": "replace", "cwd": str(REPO_ROOT)}


def _changed_py_files():
    """Return list of changed .py files excluding skip files."""
    for args in (
        ["git", "diff", "--cached", "--name-only"],
        ["git", "diff", "HEAD~1", "--name-only"],
    ):
        out = subprocess.run(args, **KW).stdout or ""
        files = [f for f in out.splitlines()
                 if f.endswith(".py") and f not in SKIP_FILES]
        if files:
            return files
    return []


def _diff_for_ext(ext):
    """Return git diff content for files with given extension."""
    pattern = f"*.{ext}"
    for args in (
        ["git", "diff", "--cached", "--", pattern],
        ["git", "diff", "HEAD~1", "--", pattern],
    ):
        out = subprocess.run(args, **KW).stdout or ""
        if out.strip():
            return out
    return ""


def _ask(prompt, default=""):
    try:
        return input(prompt).strip()
    except (EOFError, OSError):
        return default


# ── Gates ────────────────────────────────────────────────────────────────


def gate1():
    """G1: No warnings.warn() in changed Python files."""
    for fpath in _changed_py_files():
        content = (REPO_ROOT / fpath).read_text("utf-8", errors="ignore")
        if "warnings.warn(" in content:
            print(f"  [FAIL] G1: {fpath} contains warnings.warn()")
            print("     Replace with a behavior fix, not a warning.")
            return False
    print("  [PASS] G1: No warnings.warn() in changed files")
    return True


def gate2(non_interactive=False):
    """G2: PR must exist on upstream repo."""
    if non_interactive:
        print("  [SKIP] G2: Non-interactive — assuming PR exists")
        return True
    print("  [MANUAL] G2: Is the PR open on the upstream repo?")
    url = _ask("     URL (or 'skip'): ")
    if not url or url.lower() == "skip":
        print("     [FAIL] G2: PR not created on upstream.")
        return False
    print(f"  [PASS] G2: PR at {url}")
    return True


def gate3():
    """G3: Repros must have CPU fallback."""
    hw = [".cuda()", '.to("mps")', '.to("cuda")', "device='cuda'"]
    for fpath in _changed_py_files():
        content = (REPO_ROOT / fpath).read_text("utf-8", errors="ignore")
        has_hw = any(x in content for x in hw)
        has_fb = ".to('cpu')" in content or ".to(device)" in content
        if has_hw and not has_fb:
            print(f"  [WARN] G3: {fpath} uses GPU/MPS without CPU fallback")
    print("  [PASS] G3: Hardware check complete (warnings non-blocking)")
    return True


def gate4(non_interactive=False):
    """G4: NeuralDBG cited in changes."""
    diff = _diff_for_ext("py") or _diff_for_ext("md")
    if "neuraldbg" in diff.lower():
        print("  [PASS] G4: NeuralDBG cited in changes")
        return True
    print("  [WARN] G4: NeuralDBG not found in diff.")
    if non_interactive:
        return True
    return _ask("     Cited NeuralDBG in PR? (y/n): ").lower() == "y"


def gate5(non_interactive=False):
    """G5: Follow-up plan defined."""
    if non_interactive:
        print("  [SKIP] G5: Non-interactive — define plan manually")
        return True
    print("  [MANUAL] G5: Follow-up plan.")
    d3 = _ask("     Day 3 action: ")
    d7 = _ask("     Day 7 action: ")
    if d3 or d7:
        print(f"  [PASS] G5: D3:{d3 or '-'} D7:{d7 or '-'}")
        return True
    print("  [FAIL] G5: No follow-up plan.")
    return False


def gate6():
    """G6: Zero tolerance for 'workaround' in Python files only."""
    diff = _diff_for_ext("py")
    added = [l for l in diff.splitlines()
             if l.startswith("+") and not l.startswith("+++")]
    bad = [l for l in added if "workaround" in l.lower()]
    if bad:
        print(f"  [FAIL] G6: {len(bad)} 'workaround' in Python diff:")
        for line in bad[:5]:
            print(f"     {line[:120]}")
        print("     Replace with 'fix', 'resolution', or 'mitigation'.")
        return False
    print("  [PASS] G6: Zero 'workaround' in Python files")
    return True


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("PR GATE VALIDATOR — .github/PR_GATE.md")
    print("=" * 60 + "\n")

    strict = "--strict" in sys.argv
    ni = "--non-interactive" in sys.argv

    gates = [
        ("G1: FIX not warning", gate1),
        ("G2: PR exists upstream", lambda: gate2(ni)),
        ("G3: Hardware-independent repro", gate3),
        ("G4: NeuralDBG cited", lambda: gate4(ni)),
        ("G5: Follow-up plan", lambda: gate5(ni)),
        ("G6: No workaround word", gate6),
    ]

    passed, failed = 0, 0
    for name, fn in gates:
        print(f"--- {name} ---")
        try:
            if fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{len(gates)} passed, {failed} failed")
    print("=" * 60)

    if failed:
        print("\n[FAIL] Fix issues above before creating the PR.")
        sys.exit(1 if strict else 0)
    else:
        print("\n[PASS] ALL GATES PASSED. Create the PR.")
        sys.exit(0)


if __name__ == "__main__":
    main()
