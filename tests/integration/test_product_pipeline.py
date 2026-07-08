"""Product-first E2E: NeuralDBG detect -> export -> Neural-Agent remediate.

Validates BUG-002, BUG-003, BUG-004, BUG-005 are fixed by NeuralSuite
before any upstream PR is considered.

Per PR_GATE.md GATE 0.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

neuralagent = pytest.importorskip("neuralagent")
from neuraldbg import NeuralDbg  # noqa: E402
from neuralagent.bridge import load_package, remediate_from_package  # noqa: E402

pytestmark = pytest.mark.integration


def _export_and_remediate(dbg, tmp_path, config=None):
    out = tmp_path / "pkg.json"
    dbg.export_aquarium_package(str(out))
    package = load_package(out)
    return remediate_from_package(package, initial_config=config or {"lr": 1.0, "activation": "ReLU"})


class TestBUG002VarlenNanProduct:
    """BUG-002 pytorch#176793 — NaN gradients (varlen_attn pattern)."""

    def test_nan_gradient_remediated_by_neuralsuite(self, tmp_path):
        torch.manual_seed(42)
        model = nn.Linear(64, 192)

        with NeuralDbg(model) as dbg:
            dbg.record_gradient_anomaly(
                "weight",
                kind="nan",
                metadata={"source": "varlen_attn", "bug": "BUG-002/pytorch#176793"},
            )
            dbg.record_loss(float("nan"))
            hypotheses = dbg.get_causal_hypotheses()
            assert len(hypotheses) >= 1

            result = _export_and_remediate(dbg, tmp_path)
            assert result["category"] in ("gradient_explosion", "data_anomaly")
            pc = result["patched_config"]
            assert pc.get("clip_grad_norm") == 1.0 or pc.get("lr", 1.0) < 1.0 or pc.get("normalize_inputs")


class TestBUG003MpsWrongGradProduct:
    """BUG-003 pytorch#177116 — MPS wrong gradients (CPU injection)."""

    def test_exploding_grad_pattern_remediated(self, tmp_path):
        torch.manual_seed(0)
        model = nn.Linear(8, 8)

        with NeuralDbg(model) as dbg:
            dbg.record_gradient_anomaly(
                "weight",
                kind="exploding",
                metadata={"source": "mps_backend", "bug": "BUG-003/pytorch#177116"},
            )
            dbg.record_loss(1e6)
            hypotheses = dbg.get_causal_hypotheses()
            assert hypotheses

            result = _export_and_remediate(dbg, tmp_path, {"lr": 0.5, "activation": "ReLU"})
            assert result["category"] == "gradient_explosion"
            assert result["patched_config"]["clip_grad_norm"] == 1.0
            assert result["patched_config"]["lr"] == pytest.approx(0.05)


class TestBUG004SdpaExplosionProduct:
    """BUG-004 transformers#44928 — SDPA gradient explosion."""

    def test_sdpa_explosion_hypothesis_remediated(self, tmp_path):
        package = {
            "hypotheses": [
                {
                    "description": (
                        "Gradient explosion in SDPA path (BUG-004 / "
                        "huggingface/transformers#44928): bf16 collapse"
                    ),
                    "confidence": 0.92,
                }
            ],
            "events": [
                {
                    "type": "gradient_health_transition",
                    "layer": "q_proj",
                    "step": 12,
                    "from": "healthy",
                    "to": "exploding",
                    "confidence": 0.9,
                }
            ],
        }
        path = tmp_path / "bug004.json"
        path.write_text(json.dumps(package))
        result = remediate_from_package(load_package(path), initial_config={"lr": 1.0})
        assert result["category"] == "sdpa_gradient_explosion"
        assert result["patched_config"]["clip_grad_norm"] == 1.0
        assert result["patched_config"]["attn_implementation"] == "flash_attention_2"


class TestBUG005LstmSampleIndependenceProduct:
    """BUG-005 pytorch#173334 — LSTM batch pollution."""

    def test_sample_independence_event_and_remediation(self, tmp_path):
        torch.manual_seed(42)
        lstm = nn.LSTM(16, 16, batch_first=True)
        x = torch.randn(2, 4, 16)

        with NeuralDbg(lstm) as dbg:
            dbg.record_sample_independence_violation(
                layer_name="lstm",
                sample_idx=1,
                batched_has_nan=True,
                single_is_valid=True,
            )
            hypotheses = dbg.get_causal_hypotheses()
            assert any("sample independence" in h.description.lower() for h in hypotheses)

            result = _export_and_remediate(dbg, tmp_path, {"lr": 0.1, "activation": "ReLU"})
            assert result["category"] == "lstm_sample_independence"
            assert result["patched_config"].get("per_sample_inference") is True
            assert result["patched_config"]["clip_grad_norm"] == 1.0
