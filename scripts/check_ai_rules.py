#!/usr/bin/env python3
"""Validate local AI governance rule files before commit."""

from __future__ import annotations

import re
import sys
from pathlib import Path

RULE_REF_RE = re.compile(r"- \*\*(rule_[^*]+)\*\*:")


def validate_agents_rule_index(
    agents_path: Path = Path("AGENTS.md"),
    rules_dir: Path = Path("rules"),
) -> list[str]:
    errors: list[str] = []
    if not agents_path.exists():
        return [f"{agents_path} is missing"]
    if not rules_dir.exists():
        return [f"{rules_dir} is missing"]

    text = agents_path.read_text(encoding="utf-8")
    if "# AGENTS.md -- Kuro Rules Redirector" not in text:
        errors.append("AGENTS.md missing Kuro redirector header")
    if "## Rule Index" not in text:
        errors.append("AGENTS.md missing Rule Index section")

    for rule_name in RULE_REF_RE.findall(text):
        rule_path = rules_dir / f"{rule_name}.md"
        if not rule_path.exists():
            display_path = Path(rules_dir.name) / rule_path.name
            errors.append(
                f"AGENTS.md references {rule_name} but {display_path.as_posix()} is missing"
            )
    return errors


def main() -> int:
    errors = validate_agents_rule_index()
    if errors:
        print("AI rule guard failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
