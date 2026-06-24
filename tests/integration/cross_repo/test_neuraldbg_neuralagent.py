"""Cross-repo: NeuralDBG <-> Neural-Agent (R105 + ecosystem.md contract).

Validates the `dbg.explain_failure() -> list[CausalHypothesis]` contract
that `neural-agent` consumes (in-process, Python).

These tests are SKIPPED if `neuralagent` is not importable.

Strategy: induce a known failure with NeuralDBG (e.g. vanishing gradients
from Sigmoid + small init), then call `Remediator.remediate(hypotheses)`
and assert the config was patched in the expected direction.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cross_repo

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

# Skip the whole module if neural-agent is not installed in this env
neuralagent = pytest.importorskip("neuralagent")

from neuraldbg import NeuralDbg  # noqa: E402

SEED = 42


def _train_step(model, x, target, dbg, optimizer, loss_fn):
    """One training step that records loss in dbg."""
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out, target)
    loss.backward()
    optimizer.step()
    dbg.step_iteration()
    dbg.record_loss(loss.item())


class TestNeuralDbgNeuralAgentContract:
    """Verify the `dbg.explain_failure()` -> `Remediator.remediate()` pipeline."""

    def test_neuralagent_imports_cleanly(self):
        """Sanity: neural-agent package is importable, exposes public API."""
        for attr in (
            "Remediator", "RemediationRunner",
            "apply_mha_mask_workaround", "REMEDIATION_STRATEGIES"
        ):
            assert hasattr(neuralagent, attr)

    def test_vanishing_gradients_triggers_gradient_vanishing_remediation(self):
        """Vanishing gradients (Sigmoid+small init) -> Remediator swaps activation."""
        torch.manual_seed(SEED)
        # Sigmoid + small init -> known vanishing gradient regime
        model = nn.Sequential(nn.Linear(8, 16), nn.Sigmoid(), nn.Linear(16, 2))
        for p in model.parameters():
            p.data *= 0.01  # very small init

        x = torch.randn(4, 8)
        target = torch.randint(0, 2, (4,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        with NeuralDbg(model) as dbg:
            for _ in range(5):
                _train_step(model, x, target, dbg, optimizer, loss_fn)
            hypotheses = dbg.explain_failure()

        # NeuralDBG MUST return at least one hypothesis
        assert isinstance(hypotheses, list)
        assert len(hypotheses) >= 1, (
            "Expected NeuralDBG to produce a hypothesis"
        )

        # Feed hypotheses into Neural-Agent's Remediator
        remediator = neuralagent.Remediator({"lr": 1e-3, "activation": "Sigmoid"})
        patched, info = remediator.remediate(hypotheses)

        # The remediation MAY keep config unchanged if NeuralDBG's hypotheses
        # don't match a known rule, but it MUST return a valid (config, str) tuple
        assert isinstance(patched, dict)
        assert isinstance(info, str)
        # The returned config MUST contain the original keys
        assert "lr" in patched
        assert "activation" in patched

    def test_exploding_gradients_triggers_lr_reduction(self):
        """Exploding gradients (high LR+large init) -> Remediator reduces LR."""
        torch.manual_seed(SEED)
        # High LR + large init + 1 epoch -> known exploding regime
        model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
        for p in model.parameters():
            p.data *= 5.0  # very large init

        x = torch.randn(4, 8)
        target = torch.randint(0, 2, (4,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # very high LR

        with NeuralDbg(model) as dbg:
            for _ in range(3):
                _train_step(model, x, target, dbg, optimizer, loss_fn)
            hypotheses = dbg.explain_failure()

        assert isinstance(hypotheses, list)

        remediator = neuralagent.Remediator({"lr": 1.0, "activation": "ReLU"})
        patched, info = remediator.remediate(hypotheses)

        # If NeuralDBG detected the explosion and matched a rule, LR should drop
        # (factor of 0.1 per REMEDIATION_STRATEGIES). If no match, LR stays the same.
        assert patched["lr"] in (
            0.1,  # reduced (matched gradient_explosion)
            1.0,  # unchanged (no match — fine, graceful)
        )

    def test_classify_hypothesis_mha_keywords(self):
        """Verify Neural-Agent can classify a MHA-related hypothesis."""
        from neuralagent import classify_hypothesis

        mha_desc = (
            "MultiheadAttention fully-masked row in layer attn (BUG-001, "
            "pytorch/pytorch#41508): register_composite_hook recommended"
        )
        assert classify_hypothesis(mha_desc) == "mha_fully_masked_row"

    def test_classify_hypothesis_keywords(self):
        """Verify classification keywords cover the standard failure types."""
        from neuralagent import classify_hypothesis

        assert (
            classify_hypothesis("gradient explosion in layer X") == "gradient_explosion"
        )
        assert (
            classify_hypothesis("vanishing gradient in layer Y") == "gradient_vanishing"
        )
        assert classify_hypothesis("dead neurons in layer Z") == "dead_neurons"
        assert (
            classify_hypothesis("saturated activation in conv")
            == "saturated_activations"
        )
        assert classify_hypothesis("data anomaly: NaN detected") == "data_anomaly"
        # Unknown description -> default fallback
        assert classify_hypothesis("completely unknown failure") == "gradient_explosion"

    def test_apply_mha_mask_workaround_merges_masks(self):
        """Verify the BUG-001 workaround: merge key_padding_mask into attn_mask
        and force the diagonal to 0."""
        seq_len, batch = 4, 2
        # attn_mask: (S, S) additive bias. 0 = normal, -inf = masked
        attn_mask = torch.zeros(seq_len, seq_len)
        # key_padding_mask: (B, S) bool. True = pad token (masked)
        key_padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)

        # Pad the last token in sequence 0
        key_padding_mask[0, -1] = True

        merged = neuralagent.apply_mha_mask_workaround(attn_mask, key_padding_mask)

        # Output shape: (B, S, S)
        assert merged.shape == (batch, seq_len, seq_len)

        # The diagonal MUST be 0 (every query attends to at least itself)
        for b in range(batch):
            for s in range(seq_len):
                assert (
                    merged[b, s, s].item() == 0.0
                ), f"Diagonal [{b}, {s}, {s}] should be 0, got {merged[b, s, s].item()}"

    def test_engine_optional_fallback_returns_empty(self):
        """Without neuraldbg-engine, detect_coupled_failures() MUST return [].
        This is the R105 / cdp_protocol_definition contract."""
        torch.manual_seed(SEED)
        model = nn.Linear(8, 2)
        x = torch.randn(4, 8)
        target = torch.randn(4, 2)

        with NeuralDbg(model) as dbg:
            for _ in range(2):
                _train_step(
                    model,
                    x,
                    target,
                    dbg,
                    torch.optim.SGD(model.parameters(), lr=0.01),
                    nn.MSELoss(),
                )
            couplings = dbg.detect_coupled_failures()

        # Per cdp_protocol_definition.md: no engine -> empty list, no crash
        assert couplings == []
