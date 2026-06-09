"""Cross-repo: NeuralDBG <-> neuraldbg-engine (R105 + ecosystem.md contract).

Validates the optional `neuraldbg-engine` upgrade:
- Discovery: `import neuraldbg_engine` works
- Same `dbg.explain_failure()` API produces richer hypotheses when engine is loaded
- When engine is absent, fallbacks run (no crash) — per cdp_protocol_definition.md

STATUS: SKELETON — TODO
- `pip install` from GitHub Packages (private registry, requires creds)
- Skip cleanly when engine is unavailable
- Compare hypothesis richness with vs without engine
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cross_repo


def test_engine_optional_import():
    """neuraldbg MUST import cleanly when engine is absent."""
    from neuraldbg import NeuralDbg  # noqa: F401
    assert True  # If we got here, no ImportError


# def test_engine_presence_changes_hypotheses():
#     """When engine is loaded, hypotheses SHOULD be richer (more confidence, more detail)."""
#     try:
#         import neuraldbg_engine  # noqa: F401
#     except ImportError:
#         pytest.skip("neuraldbg-engine not installed")
#     # TODO: induce a known failure, compare hypothesis lists with/without engine


# def test_cdp_fallback_returns_empty_list():
#     """Per cdp_protocol_definition.md: detect_coupled_failures() without engine returns []."""
#     from neuraldbg import NeuralDbg
#     import torch.nn as nn
#
#     model = nn.Linear(10, 2)
#     with NeuralDbg(model) as dbg:
#         result = dbg.detect_coupled_failures()
#     assert result == []  # No engine = no coupling detection, no crash


def test_placeholder_skip():
    """Marker test — once the real tests are written, remove this."""
    pytest.skip("Skeleton — see TODOs in module docstring")
