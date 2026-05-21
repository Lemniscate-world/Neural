#!/usr/bin/env python3
"""Run Safety on project requirement files with UTF-8-safe output."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REQUIREMENT_FILES = [
    Path("requirements.txt"),
    Path("requirements-dev.txt"),
    Path("requirements-prod.txt"),
]


def run_safety(requirements_file: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [
        sys.executable,
        "-m",
        "safety",
        "check",
        "-r",
        str(requirements_file),
        "--output",
        "json",
    ]
    completed = subprocess.run(command, env=env, text=True)
    return completed.returncode


def main() -> int:
    missing = [str(path) for path in REQUIREMENT_FILES if not path.exists()]
    if missing:
        print(f"Missing requirement files: {', '.join(missing)}", file=sys.stderr)
        return 1

    status = 0
    for requirements_file in REQUIREMENT_FILES:
        rc = run_safety(requirements_file)
        if rc != 0:
            status = rc
    return status


if __name__ == "__main__":
    raise SystemExit(main())
