"""Cross-repo: NeuralDBG <-> Aquarium (R105 + ecosystem.md contract).

Validates the `events.json` schema that Aquarium consumes (out-of-process, JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cross_repo

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from neuraldbg import NeuralDbg  # noqa: E402


# Schema location (canonical, per COMPATIBILITY_MATRIX.md)
SCHEMA_PATH = Path(__file__).parents[3] / "neuraldbg" / "schema" / "events.json"


class TestAquariumJSONContract:
    """Verify the JSON export matches the schema Aquarium expects."""

    def test_schema_file_exists(self):
        """The schema file is the contract — it MUST exist."""
        assert SCHEMA_PATH.exists(), f"Schema missing at {SCHEMA_PATH}"
        # Should be valid JSON
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        assert schema.get("title") == "NeuralDbg Aquarium Bridge Schema"

    def test_export_aquarium_package_produces_valid_json(self, tmp_path):
        """`dbg.export_aquarium_package()` MUST produce a JSON file with the
        required top-level keys per schema."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        x = torch.randn(4, 8)
        target = torch.randint(0, 2, (4,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        out_file = tmp_path / "run.json"

        with NeuralDbg(model) as dbg:
            for _ in range(3):
                optimizer.zero_grad()
                loss = loss_fn(model(x), target)
                loss.backward()
                optimizer.step()
                dbg.step_iteration()
                dbg.record_loss(loss.item())
            dbg.export_aquarium_package(str(out_file))

        assert out_file.exists(), "Export file was not created"

        with open(out_file) as f:
            package = json.load(f)

        # Required top-level keys per schema + per test_aquarium_export.py
        for key in (
            "events",
            "hypotheses",
            "couplings",
            "first_failure_layer",
            "first_failure_step",
            "loss_history",
        ):
            assert key in package, f"Missing top-level key: {key}"

        # events list
        assert isinstance(package["events"], list)
        # loss_history should be a list of floats
        assert isinstance(package["loss_history"], list)
        assert all(isinstance(v, (int, float)) for v in package["loss_history"])
        assert len(package["loss_history"]) == 3  # we ran 3 steps

    def test_mermaid_export_returns_string(self):
        """`dbg.export_mermaid_causal_graph()` MUST return a Mermaid string."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        with NeuralDbg(model) as dbg:
            # Run a single step so there are events to graph
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(4, 8)
            target = torch.randint(0, 2, (4,))
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), target)
            loss.backward()
            optimizer.step()
            dbg.step_iteration()
            dbg.record_loss(loss.item())
            graph = dbg.export_mermaid_causal_graph()

        assert isinstance(graph, str)
        # Mermaid graph types: flowchart, sequenceDiagram, etc.
        assert any(
            kw in graph
            for kw in ("flowchart", "graph", "sequenceDiagram", "graph TD", "graph LR")
        ), f"Output does not look like Mermaid: {graph[:200]!r}"
