#!/usr/bin/env python3
"""Run Bandit on tracked Python source files only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXCLUDED_PREFIXES = (
    "tests/",
    ".venv/",
    "venv/",
    "htmlcov/",
    ".pytest_cache/",
    "dist/",
    "build/",
)


def tracked_python_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "*.py"], text=True, encoding="utf-8"
    )
    files = []
    for raw_path in output.splitlines():
        path = raw_path.replace("\\", "/")
        if path.startswith(EXCLUDED_PREFIXES):
            continue
        if any(part == "__pycache__" for part in Path(path).parts):
            continue
        files.append(path)
    return files


def main() -> int:
    files = tracked_python_files()
    if not files:
        print("No tracked Python source files to scan.")
        return 0
    command = [
        sys.executable,
        "-m",
        "bandit",
        "-ll",
        "--skip",
        "B101",
        *files,
    ]
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
