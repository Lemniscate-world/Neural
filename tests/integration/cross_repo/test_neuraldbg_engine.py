"""Cross-repo: NeuralDBG <-> neuraldbg-engine (R105 + ecosystem.md contract).

Validates the optional `neuraldbg-engine` upgrade:
- Discovery: `import neuraldbg_engine` works
- Same `dbg.explain_failure()` API works whether or not the engine is loaded
- When engine is absent, the lightweight fallbacks in core run (no crash)

If `neuraldbg_engine` is not installed, the tests for the engine are skipped
and only the fallback contract is tested.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cross_repo

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

# Probe for the optional engine (not required)
try:
    import neuraldbg_engine  # noqa: F401

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


from neuraldbg import NeuralDbg  # noqa: E402


def _train_step(model, x, target, dbg, optimizer, loss_fn):
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    dbg.step_iteration()
    dbg.record_loss(loss.item())


class TestEngineDiscovery:
    """The `import neuraldbg_engine` line MUST be tolerant."""

    def test_neuraldbg_imports_without_engine(self):
        """Core MUST import cleanly when engine is absent (cdp_protocol_definition)."""
        from neuraldbg import _HAS_ENGINE, NeuralDbg  # type: ignore

        # If engine is installed, _HAS_ENGINE is True; else False.
        # Either way, NeuralDbg is importable.
        assert NeuralDbg is not None
        assert isinstance(_HAS_ENGINE, bool)

    def test_neuraldbg_constructor_with_or_without_engine(self):
        """NeuralDbg(model) MUST work whether or not the engine is present."""
        torch.manual_seed(42)
        model = nn.Linear(8, 2)
        dbg = NeuralDbg(model)
        # If engine is present, dbg._causal_engine is set
        if HAS_ENGINE:
            assert dbg._causal_engine is not None
        else:
            assert dbg._causal_engine is None


class TestFallbackContract:
    """When the engine is absent, the fallbacks MUST behave per cdp_protocol."""

    def test_detect_coupled_failures_returns_empty_without_engine(self):
        """Per cdp_protocol_definition.md: detect_coupled_failures() w/o engine returns []."""
        if HAS_ENGINE:
            pytest.skip("Engine is installed — this test validates the fallback path only")
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        x = torch.randn(4, 8)
        target = torch.randint(0, 2, (4,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with NeuralDbg(model) as dbg:
            for _ in range(3):
                _train_step(model, x, target, dbg, optimizer, loss_fn)
            result = dbg.detect_coupled_failures()

        assert result == [], f"Expected fallback to return [], got {result!r}"

    def test_explain_failure_returns_list_without_engine(self):
        """explain_failure() MUST work without the engine (may return [] or
        a basic hypothesis, but no crash)."""
        if HAS_ENGINE:
            pytest.skip("Engine is installed — this test validates the fallback path only")
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 2))
        x = torch.randn(4, 8)
        target = torch.randint(0, 2, (4,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with NeuralDbg(model) as dbg:
            for _ in range(2):
                _train_step(model, x, target, dbg, optimizer, loss_fn)
            # MUST NOT raise
            hypotheses = dbg.explain_failure()

        assert isinstance(hypotheses, list)


@pytest.mark.skipif(not HAS_ENGINE, reason="neuraldbg-engine not installed (closed beta)")
class TestEngineRichness:
    """When the engine IS installed, hypotheses SHOULD be richer / more confident."""

    def test_engine_produces_hypotheses(self):
        """With the engine, explain_failure() should produce a non-empty list."""
        torch.manual_seed(42)
        # Setup a model that will clearly fail
        model = nn.Sequential(nn.Linear(8, 16), nn.Sigmoid(), nn.Linear(16, 2))
        for p in model.parameters():
            p.data *= 0.001  # very small init -> vanishing

        x = torch.randn(4, 8)
        target = torch.randint(0, 2, (4,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        with NeuralDbg(model) as dbg:
            for _ in range(3):
                _train_step(model, x, target, dbg, optimizer, loss_fn)
            hypotheses = dbg.explain_failure()

        assert isinstance(hypotheses, list)
        # With the engine, we expect at least one hypothesis
        assert len(hypotheses) >= 1, "Engine-loaded NeuralDbg should produce hypotheses"

    def test_engine_api_contract(self):
        """Verify the CausalEngine public API matches COMPATIBILITY_MATRIX.md."""
        from neuraldbg_engine import CausalEngine  # type: ignore

        torch.manual_seed(42)
        model = nn.Linear(8, 2)
        with NeuralDbg(model) as dbg:
            engine = CausalEngine(dbg)
            # Per COMPATIBILITY_MATRIX.md, these methods MUST exist
            assert hasattr(engine, "detect_gradient_transition")
            assert hasattr(engine, "classify_gradient_health")
            assert hasattr(engine, "classify_activation_health")
            # Smoke test: call with sane inputs
            assert engine.classify_gradient_health(1.0) in (
                "healthy",
                "stable",
                "vanishing",
                "exploding",
            )
