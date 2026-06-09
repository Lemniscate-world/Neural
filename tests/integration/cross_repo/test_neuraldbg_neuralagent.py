"""Cross-repo: NeuralDBG <-> Neural-Agent (R105 + ecosystem.md contract).

Validates the `dbg.explain_failure() -> list[CausalHypothesis]` contract
that `neural-agent` consumes (in-process, Python).

STATUS: SKELETON — TODO
- Cloning logic for the sibling `Neural-Agent` repo
- Discovery of the `neuralagent` package (editable install in CI)
- E2E test: induce a failure with NeuralDBG, run RemediationRunner,
  verify that the patched training loop no longer fails.
"""

from __future__ import annotations

import pytest

# Markers
pytestmark = pytest.mark.cross_repo


# --- Test fixtures (TODO) ---

# @pytest.fixture(scope="session")
# def neuralagent_installed() -> bool:
#     """Skip all tests in this module if neural-agent is not installed."""
#     try:
#         import neuralagent  # noqa: F401
#         return True
#     except ImportError:
#         return False


# --- Tests (TODO) ---

# def test_neuralagent_imports_neuraldbg(neuralagent_installed):
#     """neural-agent MUST depend on neuraldbg (per ecosystem.md dependency direction)."""
#     if not neuralagent_installed:
#         pytest.skip("neural-agent not installed")
#     import neuralagent
#     import neuraldbg
#     assert neuralagent.__depends_on__neuraldbg__ is True  # sentinel


# def test_mha_mask_workaround_via_neuralagent(neuralagent_installed):
#     """E2E: MHA fully-masked-row -> apply_mha_mask_workaround() fixes it."""
#     if not neuralagent_installed:
#         pytest.skip("neural-agent not installed")
#     import torch
#     import torch.nn as nn
#     from neuraldbg import NeuralDbg
#     from neuralagent import RemediationRunner
#     from neuralagent.remediation_rules import apply_mha_mask_workaround
#
#     model = nn.MultiheadAttention(d_model=64, num_heads=4)
#     with NeuralDbg(model) as dbg:
#         runner = RemediationRunner(dbg)
#         runner.register(apply_mha_mask_workaround)
#         # ... induce failure, verify fix ...


def test_placeholder_skip():
    """Marker test — once the real tests are written, remove this."""
    pytest.skip("Skeleton — see TODOs in module docstring")
