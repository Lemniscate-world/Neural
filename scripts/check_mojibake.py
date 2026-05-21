#!/usr/bin/env python3
"""Reject common mojibake sequences in tracked text files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

BROKEN_SEQUENCES = (
    "\u00e2\u20ac\u201d",
    "\u00e2\u20ac\u201c",
    "\u00e2\u20ac\u0153",
    "\u00e2\u20ac\ufffd",
    "\u00e2\u20ac\u02dc",
    "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u00a6",
    "\u00e2\u2020\u2019",
    "\u00e2\u0153\u2026",
    "\u00e2\ufffd\u0152",
    "\u00f0\u0178",
    "\u00c3\u00a9",
    "\u00c3\u00a8",
    "\u00c3\u00aa",
    "\u00c3\u00a0",
    "\u00c3\u00a2",
    "\u00c3\u00a7",
    "\u00c3\u2030",
    "\u00c3\u20ac",
    "\ufffd",
)

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_text_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True, encoding="utf-8")
    paths = []
    for raw_path in output.splitlines():
        path = Path(raw_path)
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            "AGENTS.md",
            "AI_GUIDELINES.md",
            "Makefile",
        }:
            paths.append(path)
    return paths


def find_mojibake(paths: Iterable[Path]) -> list[tuple[Path, int, str]]:
    findings: list[tuple[Path, int, str]] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            findings.append((path, 0, "invalid utf-8"))
            continue
        for line_number, line in enumerate(lines, start=1):
            for sequence in BROKEN_SEQUENCES:
                if sequence in line:
                    findings.append((path, line_number, sequence))
                    break
    return findings


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else tracked_text_files()
    findings = find_mojibake(paths)
    if findings:
        print("Mojibake detected. Fix encoding before committing:", file=sys.stderr)
        for path, line_number, sequence in findings:
            print(f"  {path}:{line_number}: {sequence}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
