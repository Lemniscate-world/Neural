"""Tests for pre-commit governance guard scripts."""

from pathlib import Path


def test_mojibake_guard_detects_broken_sequences(tmp_path):
    from scripts.check_mojibake import find_mojibake

    bad = tmp_path / "bad.md"
    good = tmp_path / "good.md"
    bad.write_text("RULE 1: Read Rules First â€” MANDATORY", encoding="utf-8")
    good.write_text("RULE 1: Read Rules First - MANDATORY", encoding="utf-8")

    findings = find_mojibake([bad, good])

    assert findings == [(bad, 1, "â€”")]


def test_ai_rule_guard_requires_indexed_rule_files(tmp_path):
    from scripts.check_ai_rules import validate_agents_rule_index

    agents = tmp_path / "AGENTS.md"
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    agents.write_text(
        "\n".join(
                [
                    "# AGENTS.md -- Kuro Rules Redirector",
                    "## Rule Index",
                    "- **rule_01_foundation**: RULE 1",
                    "- **rule_101_tensor_and_pytest_safety**: RULE 101",
            ]
        ),
        encoding="utf-8",
    )
    (rules_dir / "rule_01_foundation.md").write_text("# RULE 1", encoding="utf-8")

    errors = validate_agents_rule_index(agents, rules_dir)

    assert errors == [
        "AGENTS.md references rule_101_tensor_and_pytest_safety but "
        "rules/rule_101_tensor_and_pytest_safety.md is missing"
    ]


def test_sensitive_file_guard_can_scan_tracked_files():
    from scripts.check_sensitive_files import is_protected

    protected_paths = [
        "SESSION_SUMMARY.md",
        "docs/tracking/acquisition_tracker.md",
        "infrastructure_planning/dvc_workflow.md",
        ".bandit",
    ]

    for path in protected_paths:
        assert is_protected(path), path

    assert not is_protected("neuraldbg/__init__.py")
