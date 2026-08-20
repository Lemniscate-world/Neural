#!/usr/bin/env python3
"""
CI Watchdog Agent — surveille et corrige la CI NeuralDBG en local.

Usage:
  python scripts/ci_agent.py              # check ponctuel local + remote
  python scripts/ci_agent.py --fix        # tente auto-fix (coverage, debounce)
  python scripts/ci_agent.py --loop 300   # boucle toutes les 300s (5min)
  python scripts/ci_agent.py --loop 300 --fix  # agent autonome

Reproduit exactement .github/workflows/ci.yml:71 + pyproject.toml:97
  pytest tests/unit --cov=neuraldbg --cov-fail-under=75

Remote: gh run list --workflow ci.yml + gh run view --log-failed (si gh CLI dispo)
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
COV_THRESHOLD = 75

# Patterns auto-fixables connus (issus du fix 3104a05a)
KNOWN_FIXES = {
    "distribution_shift": "debounce ≥2 steps (engine/data.py:98) — patch test_data_anomaly.py",
    "SATURATED": "P2b: gradient HEALTHY remplace SATURATED (engine/gradient.py:20)",
    "coverage.*fail-under": "ajouter tests smoke wandb/lightning/rl_detector (test_integrations_smoke.py)",
    "torchvision::nms": "skip Windows (torchvision DLL)",
}

def run(cmd, **kw):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT, **kw)  # nosec B602 - intentional local gh/pytest invocation
        return r.returncode, r.stdout + r.stderr
    except Exception as e:
        return 1, str(e)

def check_local(fix=False):
    print(f"\n{'='*60}")
    print(f"[{datetime.now().isoformat()}] LOCAL CHECK — pytest --cov-fail-under={COV_THRESHOLD}")
    print(f"{'='*60}")
    # Run with coverage gate (reproduit ci.yml:75 + pyproject.toml:97)
    code2, out2 = run(f'"{sys.executable}" -m pytest tests/unit -q --cov=neuraldbg --cov-fail-under={COV_THRESHOLD} --tb=line')
    combined = out2

    # Parse coverage
    m = re.search(r"Total coverage:\s*([\d.]+)%", combined)
    cov = float(m.group(1)) if m else None
    if cov is not None:
        status = "PASS" if cov >= COV_THRESHOLD else "FAIL"
        print(f"Coverage: {cov:.2f}% [{status}] (gate {COV_THRESHOLD}%)")

    # Parse failures
    failed = re.findall(r"FAILED\s+(\S+)", combined)
    skipped = len(re.findall(r"SKIPPED", combined))
    passed = re.findall(r"(\d+) passed", combined)
    if failed:
        print(f"Tests: {len(failed)} FAILED, {skipped} skipped")
        for f in failed[:10]:
            print(f"  - {f}")
    else:
        print(f"Tests: {passed[0] if passed else '?'} passed, {skipped} skipped — PASS")

    # Coverage table
    if "Name" in combined and "Stmts" in combined:
        for line in combined.splitlines():
            if "TOTAL" in line:
                print(line.strip())

    if fix and (failed or (cov and cov < COV_THRESHOLD)):
        print("\n[AGENT] --fix activé → suggestions:")
        if cov and cov < COV_THRESHOLD:
            print("  - Coverage <75%: ajouter tests pour neuraldbg/integrations/*, neuraldbg/rl_detector.py")
            print("    → voir tests/unit/test_integrations_smoke.py:1 (modèle)")
        for f in failed:
            for pat, hint in KNOWN_FIXES.items():
                if pat.lower() in f.lower() or pat.lower() in combined.lower():
                    print(f"  - {f}: {hint}")
                    break
        print("  (auto-patch complet nécessite édition manuelle — voir commit 3104a05a)")
        return False

    ok = (not failed) and (cov is None or cov >= COV_THRESHOLD)
    print(f"\nLOCAL: {'PASS' if ok else 'FAIL'}")
    return ok

def check_remote():
    print(f"\n{'='*60}")
    print(f"[{datetime.now().isoformat()}] REMOTE CHECK — gh run list --workflow ci.yml")
    print(f"{'='*60}")
    code, out = run("gh run list --workflow ci.yml --limit 5")
    if code != 0 or "gh" in out.lower() and "not found" in out.lower():
        print("gh CLI indisponible ou non authentifié — skip remote")
        print(out[:500])
        return None
    print(out[:2000])
    # Cherche dernier run failed
    if "failure" in out.lower():
        print("\n[REMOTE] Dernier run en échec détecté — fetch log...")
        # Extrait run id du premier échec
        m = re.search(r"(\d{10,})", out)
        if m:
            run_id = m.group(1)
            c2, out2 = run(f"gh run view {run_id} --log-failed 2>&1 | head -n 80")
            # Windows: pas de head, on coupe
            print(out2[:3000])
            # Parse FAILED dans log distant
            f2 = re.findall(r"FAILED\s+(\S+)", out2)
            if f2:
                print(f"\nRemote failures: {len(f2)}")
                for f in f2[:5]:
                    print(f"  - {f}")
            if "Coverage failure" in out2:
                m2 = re.search(r"total of (\d+)", out2)
                if m2:
                    print(f"Remote coverage: {m2.group(1)}% < 75%")
            return False
    elif "completed" in out.lower() and "success" in out.lower():
        print("\nREMOTE: ✅ dernier run success")
        return True
    return None

def main():
    ap = argparse.ArgumentParser(description="CI Watchdog Agent")
    ap.add_argument("--fix", action="store_true", help="tente suggestions auto-fix")
    ap.add_argument("--loop", type=int, default=0, help="boucle toutes les N secondes (0 = ponctuel)")
    ap.add_argument("--remote-only", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    def once():
        ok_local = None
        ok_remote = None
        if not args.remote_only:
            ok_local = check_local(fix=args.fix)
        if not args.local_only:
            ok_remote = check_remote()
        # Resume
        print(f"\n{'='*60}")
        print("RESUME")
        if ok_local is not None:
            print(f"  Local:  {'PASS' if ok_local else 'FAIL'}")
        if ok_remote is not None:
            print(f"  Remote: {'PASS' if ok_remote else 'FAIL' if ok_remote is not None else 'UNKNOWN'}")
        print(f"{'='*60}\n")
        return ok_local and (ok_remote is None or ok_remote is True)

    if args.loop > 0:
        print(f"[AGENT] Loop toutes les {args.loop}s — Ctrl+C pour arreter (fix={args.fix})")
        try:
            while True:
                once()
                print(f"[AGENT] sleep {args.loop}s...")
                time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\n[AGENT] arrete")
    else:
        ok = once()
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
