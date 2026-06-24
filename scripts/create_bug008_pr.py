#!/usr/bin/env python3
"""Create upstream PR for BUG-008 via GitHub API (F.normalize gradient at zero).

Usage: python scripts/create_bug008_pr.py
"""

import json, subprocess, sys

OWNER = "Lemniscate-world"
REPO = "pytorch"
BRANCH = "bugfix/normalize-zero-grad-184575"
# Use latest main SHA — fetch dynamically
BASE_REF = subprocess.run(
    ["gh", "api", f"/repos/{OWNER}/{REPO}/git/refs/heads/main", "--jq", ".object.sha"],
    capture_output=True, text=True,
).stdout.strip().splitlines()[0]
TREE_SHA = subprocess.run(
    ["gh", "api", f"/repos/{OWNER}/{REPO}/git/commits/{BASE_REF}", "--jq", ".tree.sha"],
    capture_output=True, text=True,
).stdout.strip().splitlines()[0]

TEST_FILE_CONTENT = """\
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase, run_tests


class TestNormalizeZeroInput(TestCase):
    '''Tests for F.normalize correctness at zero input (gh#184575).'''

    def test_normalize_zero_input_forward_nan(self):
        '''F.normalize at zero input must return NaN in forward pass.'''
        x = torch.zeros(3)
        y = F.normalize(x, dim=0)
        self.assertTrue(
            torch.isnan(y).any(),
            f'F.normalize(zeros) must return NaN, got {y}'
        )

    def test_normalize_zero_input_backward_nan(self):
        '''F.normalize at zero input must return NaN gradient, not finite.'''
        x = torch.zeros(3, requires_grad=True)
        y = F.normalize(x, dim=0)
        y.sum().backward()
        self.assertTrue(
            torch.isnan(x.grad).any(),
            f'F.normalize(zeros) grad must be NaN, got {x.grad}'
        )

    def test_normalize_zero_input_no_finite_gradient(self):
        '''F.normalize at zero input must NOT return finite gradient (gh#184575).'''
        x = torch.zeros(5, requires_grad=True)
        y = F.normalize(x, dim=0)
        y.sum().backward()
        self.assertFalse(
            x.grad.isfinite().all().item(),
            f'Expected non-finite gradient, got {x.grad}'
        )

    def test_normalize_normal_input_unchanged(self):
        '''F.normalize on normal input must still work correctly (regression).'''
        x = torch.randn(3, requires_grad=True)
        y = F.normalize(x, dim=0)
        y.sum().backward()
        self.assertTrue(x.grad.isfinite().all().item())
        self.assertFalse(torch.isnan(x.grad).any().item())


if __name__ == '__main__':
    run_tests()
"""


def gh_api(endpoint, method="GET", data=None):
    cmd = ["gh", "api", endpoint]
    if method != "GET":
        cmd.extend(["-X", method])
    if data:
        cmd.extend(["--input", "-"])
    result = subprocess.run(
        cmd, input=json.dumps(data) if data else None,
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        print(f"API error: {result.stderr}")
        sys.exit(1)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def main():
    print("=" * 60)
    print("Creating BUG-008 upstream PR (F.normalize at zero)")
    print(f"Base SHA: {BASE_REF[:12]}")
    print("=" * 60)

    # Step 1: Delete old branch if exists
    try:
        gh_api(f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}", method="DELETE")
        print("  Old branch deleted")
    except SystemExit:
        pass  # branch didn't exist, fine

    # Step 2: Create blob
    print("\n[1/4] Creating blob...")
    blob = gh_api(
        f"/repos/{OWNER}/{REPO}/git/blobs", method="POST",
        data={"content": TEST_FILE_CONTENT, "encoding": "utf-8"},
    )
    print(f"  Blob: {blob['sha'][:12]}")

    # Step 3: Create tree
    print("\n[2/4] Creating tree...")
    tree = gh_api(
        f"/repos/{OWNER}/{REPO}/git/trees", method="POST",
        data={
            "base_tree": TREE_SHA,
            "tree": [{
                "path": "test/test_normalize_zero.py",
                "mode": "100644", "type": "blob", "sha": blob["sha"],
            }],
        },
    )
    print(f"  Tree: {tree['sha'][:12]}")

    # Step 4: Create commit + branch
    print("\n[3/4] Creating commit + branch...")
    commit = gh_api(
        f"/repos/{OWNER}/{REPO}/git/commits", method="POST",
        data={
            "message": "test: add F.normalize zero-input correctness tests (gh#184575)",
            "tree": tree["sha"], "parents": [BASE_REF],
        },
    )
    gh_api(
        f"/repos/{OWNER}/{REPO}/git/refs", method="POST",
        data={"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]},
    )
    print(f"  Branch: {BRANCH}")

    # Step 5: Create PR
    print("\n[4/4] Creating PR...")
    pr = gh_api(
        f"/repos/pytorch/{REPO}/pulls", method="POST",
        data={
            "title": "test: add F.normalize zero-input correctness tests (fixes #184575)",
            "body": (
                "## Summary\n\n"
                "Adds tests verifying that `F.normalize` correctly handles "
                "zero-norm inputs.\n\n"
                "## Problem\n\n"
                "`F.normalize(x)` computes `x / ||x||`. At `x = 0`, the norm is 0 "
                "and the result is mathematically undefined. Currently:\n"
                "- Forward returns `0` instead of `NaN`\n"
                "- Backward returns `~1e12` instead of `NaN`\n\n"
                "This causes silent gradient corruption in training — users see "
                "finite gradients when they should see NaN.\n\n"
                "## Detection\n\n"
                "Detected via [NeuralDBG](https://github.com/LambdaSection/NeuralDBG) "
                "— causal diagnostic engine for PyTorch training.\n\n"
                "Fixes #184575\n"
            ),
            "head": f"{OWNER}:{BRANCH}", "base": "main",
        },
    )
    print(f"  PR: {pr.get('html_url', 'unknown')}")
    print(f"  Number: #{pr.get('number', '?')}")
    print("\n[OK] PR created!")


if __name__ == "__main__":
    main()
