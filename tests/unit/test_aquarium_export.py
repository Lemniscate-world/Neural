"""Tests for Aquarium JSON export functionality."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neuraldbg import NeuralDbg

SEED = 42


class TestAquariumExportSchema:
    """Verify the exported JSON matches the Aquarium schema."""

    REQUIRED_TOP_KEYS = {
        "events",
        "hypotheses",
        "couplings",
        "first_failure_layer",
        "first_failure_step",
        "loss_history",
    }

    REQUIRED_EVENT_KEYS = {"type", "layer", "step", "to", "confidence"}

    def _run_training(self, model, num_steps=5):
        x = torch.randn(8, 10)
        y = torch.randint(0, 2, (8,))
        data = DataLoader(TensorDataset(x, y), batch_size=8)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        with NeuralDbg(model) as dbg:
            for step in range(num_steps):
                for bx, by in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    loss = nn.CrossEntropyLoss()(model(bx), by)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break
        return dbg

    def _export_and_load(self, dbg, tmp_path):
        out = tmp_path / "aquarium_export"
        path = dbg.export_aquarium_package(str(out))
        assert Path(path).exists()
        with open(path) as f:
            return json.load(f)

    def test_top_level_keys_present(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        missing = self.REQUIRED_TOP_KEYS - set(pkg.keys())
        assert not missing, f"Missing required keys: {missing}"

    def test_events_is_list(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        assert isinstance(pkg["events"], list)

    def test_hypotheses_is_list(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        assert isinstance(pkg["hypotheses"], list)

    def test_couplings_is_list(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        assert isinstance(pkg["couplings"], list)

    def test_first_failure_layer_is_dict(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        assert isinstance(pkg["first_failure_layer"], dict)

    def test_first_failure_step_is_dict(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        assert isinstance(pkg["first_failure_step"], dict)

    def test_loss_history_is_list(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        assert isinstance(pkg["loss_history"], list)
        assert len(pkg["loss_history"]) == 5  # 5 steps

    def test_loss_history_values_match(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        assert pkg["loss_history"] == dbg.loss_history

    def test_event_has_required_fields(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        for event in pkg["events"]:
            missing = self.REQUIRED_EVENT_KEYS - set(event.keys())
            assert not missing, f"Event missing keys: {missing}"

    def test_event_types_are_strings(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        for event in pkg["events"]:
            assert isinstance(event["type"], str)
            assert isinstance(event["layer"], str)
            assert isinstance(event["step"], int)
            assert isinstance(event["confidence"], (int, float))

    def test_export_with_nan_triggers_events(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        x = torch.tensor(float("nan")).expand(4, 10)
        y = torch.randint(0, 2, (4,))
        data = DataLoader(TensorDataset(x, y), batch_size=4)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        with NeuralDbg(model) as dbg:
            for step in range(3):
                for bx, by in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    loss = nn.CrossEntropyLoss()(model(bx), by)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break
        pkg = self._export_and_load(dbg, tmp_path)
        assert len(pkg["events"]) > 0
        assert len(pkg["loss_history"]) == 3

    def test_export_returns_path(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        out = tmp_path / "test_return"
        result = dbg.export_aquarium_package(str(out))
        assert result == str(out)
        assert Path(result).exists()

    def test_export_json_is_valid(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        dbg = self._run_training(model)
        pkg = self._export_and_load(dbg, tmp_path)
        # Re-serialize and parse to ensure it's valid JSON
        re_parsed = json.loads(json.dumps(pkg))
        assert re_parsed == pkg

    def test_export_deep_model(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        dbg = self._run_training(model, num_steps=3)
        pkg = self._export_and_load(dbg, tmp_path)
        missing = self.REQUIRED_TOP_KEYS - set(pkg.keys())
        assert not missing
        for event in pkg["events"]:
            assert "layer" in event
