#!/usr/bin/env python3
"""Pre-commit hook script to prevent committing sensitive/private files.

This script checks the files staged for commit against the protected patterns
defined in Rule 76 (and general project security requirements).
"""

import fnmatch
import subprocess
import sys

# Protected patterns from Rule 76
PROTECTED_PATTERNS = [
    # Specific file names (exact match or pattern)
    "PLAN.md",
    "ROADMAP.md",
    "PRD.md",
    "ideas.md",
    "moat.md",
    "SESSION_SUMMARY.md",
    "SESSION_SUMMARY.md.bak",
    "SESSION_SUMMARY.docx",
    "SESSION_SUMMARY.docx.bak",
    "acquisition_tracker.md",
    "decision-memo.md",
    "LAUNCH_POSTS.md",
    "PROJECTS.md",
    "credentials.json",
    # Extensions / Wildcards
    "*.pem",
    "*.key",
    "*.p12",
    ".env",
    ".env.*",
    "service_account*.json",
    "bandit*.json",
    "bandit*.txt",
    "safety-report.json",
    "trivy*.json",
    "snyk*.json",
    ".coverage",
    ".coverage.*",
    # Folders (any file inside)
    "concept/*",
    "secrets/*",
    "prompts/*",
    "research/*",
    "infrastructure_planning/*",
    "plans/*",
    "htmlcov/*",
    ".antigravity/*",
    ".cursor/*",
    # Docs paths
    "docs/launch_plan_*.md",
    "docs/hn_feedback_log.md",
    "docs/community_post_template.md",
    "docs/launch_postmortem.md",
    "docs/cdp_protocol_definition.md",
    "docs/STRUCTURE_DOCS.md",
    "docs/verification_report_*.md",
    "docs/tracking/*",
    "docs/architecture/GAD.md",
    "docs/strategy/moat.md",
    "docs/v0_landing_page_prompt.md",
    "docs/competition/competitive_analysis.md",
    "docs/competition/desk_research.md",
    "docs/guides/AI_GUIDELINES.md",
    "interview_collection_guide.md",
    "mom_test_template.md",
]


def get_staged_files():
    """Get the list of files currently staged for commit."""
    try:
        output = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], text=True
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
    except subprocess.CalledProcessError as e:
        print(f"Error checking staged files: {e}", file=sys.stderr)
        return []


def is_protected(filepath):
    """Check if the given filepath matches any protected pattern."""
    # Normalize path to forward slashes for cross-platform pattern matching
    norm_path = filepath.replace("\\", "/")

    for pattern in PROTECTED_PATTERNS:
        # Direct pattern match
        if fnmatch.fnmatchcase(norm_path, pattern):
            return True
        # Match pattern anywhere in the path (e.g. prompts/* matches subfolders)
        if fnmatch.fnmatchcase(norm_path, f"*/{pattern}"):
            return True
        # Check folder prefixes
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            if norm_path.startswith(prefix) or f"/{prefix}" in norm_path:
                return True
    return False


def main():
    staged_files = get_staged_files()
    blocked_files = []

    for filepath in staged_files:
        if is_protected(filepath):
            blocked_files.append(filepath)

    if blocked_files:
        print("=" * 70, file=sys.stderr)
        print(
            "ERROR: Staged files contain sensitive/private files protected by Rule 76!",
            file=sys.stderr,
        )
        print(
            "The commit has been blocked to prevent private data leaks.",
            file=sys.stderr,
        )
        print("-" * 70, file=sys.stderr)
        print("Blocked files:", file=sys.stderr)
        for filepath in blocked_files:
            print(f"  - {filepath}", file=sys.stderr)
        print("-" * 70, file=sys.stderr)
        print("To unstage these files, run:", file=sys.stderr)
        print("  git restore --staged <file>", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
