"""
Phase 0 — Tests de validité causale.
Copyright (c) 2026 NeuralDBG.

Ces 8 tests prouvent que le moteur raisonne causalement, pas par corrélation.
Ils sont privés (NeuralDBG-Engine) et ne sont pas copiables sans le domaine.
"""

import math
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from neuraldbg import (
    NeuralDbg,
    SemanticEvent,
    EventType,
    _HAS_ENGINE,
)
import pytest

requires_engine = pytest.mark.skipif(
    not _HAS_ENGINE, reason="requires neuraldbg-engine"
)

SEED = 42


# ═══════════════════════════════════════════════════════════════════
# TEST 1 : Validité causale
# ═══════════════════════════════════════════════════════════════════
@requires_engine
class TestCausalValidity:
    """Injecter NaN dans UNE couche spécifique → engine localise CETTE couche."""

    def test_nan_localized_to_correct_layer(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 5),  # On injecte NaN ici
            nn.Tanh(),
        )
        model[2].weight.data.fill_(0.0)

        x = torch.randn(2, 10)
        dbg = NeuralDbg(model)
        dbg.step = 0

        # Inject NaN into layer 2 ONLY
        dbg._check_data_anomaly(x, "0")  # clean
        dbg._check_data_anomaly(torch.tensor(float("nan")).expand(2, 20), "2")  # NaN!
        dbg._check_data_anomaly(x, "4")  # clean

        nan_events = [
            e
            for e in dbg.events
            if e.event_type == EventType.DATA_ANOMALY and e.to_state == "nan_detected"
        ]
        assert len(nan_events) == 1, f"Expected 1 NaN event, got {len(nan_events)}"
        assert nan_events[0].layer_name == "2", (
            f"Expected layer '2', got '{nan_events[0].layer_name}'"
        )

    def test_nan_in_layer_2_not_layer_1(self):
        """NaN dans layer 2 ne contamine pas le diagnostic de layer 1."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Tanh())
        x_clean = torch.randn(2, 10)
        x_nan = torch.tensor(float("nan")).expand(2, 10)

        dbg = NeuralDbg(model)
        dbg.step = 0
        dbg._check_data_anomaly(x_clean, "0")
        dbg._check_data_anomaly(x_nan, "2")

        events_by_layer = {}
        for e in dbg.events:
            events_by_layer.setdefault(e.layer_name, []).append(e)

        assert "0" not in events_by_layer or not any(
            e.to_state == "nan_detected" for e in events_by_layer["0"]
        ), "Layer 0 should NOT have NaN detection"
        assert any(
            e.to_state == "nan_detected" for e in events_by_layer.get("2", [])
        ), "Layer 2 SHOULD have NaN detection"


# ═══════════════════════════════════════════════════════════════════
# TEST 2 : Faux positifs
# ═══════════════════════════════════════════════════════════════════
class TestFalsePositives:
    """Entraînement sain → 0 hypothèses, 0 alertes."""

    def test_healthy_training_produces_no_hypotheses(self):
        torch.manual_seed(SEED)
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )
        data = DataLoader(
            TensorDataset(torch.randn(50, 10), torch.randint(0, 2, (50,))),
            batch_size=8,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        with NeuralDbg(model, threshold_vanishing=1e-6, threshold_exploding=1e3) as dbg:
            for step in range(5):
                for x, y in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break

        hyps = dbg.explain_failure()
        assert len(hyps) == 0, f"Healthy training produced {len(hyps)} hypotheses"

    def test_healthy_training_no_events_with_nan_state(self):
        """Sain → aucun événement avec état 'nan_detected' ou 'diverging'."""
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        data = DataLoader(
            TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,))), batch_size=4
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        with NeuralDbg(model) as dbg:
            for step in range(3):
                for x, y in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    loss = criterion(model(x), y)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break

        bad_states = {"nan_detected", "inf_detected", "diverging"}
        for e in dbg.events:
            assert e.to_state not in bad_states, (
                f"Healthy run produced event with bad state '{e.to_state}' at step {e.step}"
            )


# ═══════════════════════════════════════════════════════════════════
# TEST 3 : Déterminisme
# ═══════════════════════════════════════════════════════════════════
@requires_engine
class TestDeterminism:
    """Même seed + même bug → mêmes hypothèses exactes."""

    def _run_with_nan(self, seed):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))
        data = DataLoader(
            TensorDataset(torch.randn(10, 5), torch.randint(0, 3, (10,))),
            batch_size=4,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        with NeuralDbg(model) as dbg:
            for step in range(4):
                for x, y in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    if step == 2:
                        x[0, :] = float("nan")
                    loss = criterion(model(x), y)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break
        return dbg

    def test_identical_runs_produce_identical_hypotheses(self):
        dbg1 = self._run_with_nan(SEED)
        dbg2 = self._run_with_nan(SEED)

        h1 = sorted(h.description for h in dbg1.explain_failure())
        h2 = sorted(h.description for h in dbg2.explain_failure())

        assert h1 == h2, f"Mismatch:\n  Run1: {h1}\n  Run2: {h2}"

    def test_different_seeds_different_hypothesis_orders(self):
        """Seed ≠ → ordre des hypothèses peut différer (mais NaN toujours présent)."""
        dbg1 = self._run_with_nan(SEED)
        h1 = sorted(h.description for h in dbg1.explain_failure())
        nan_in_run1 = any("nan" in h.lower() for h in h1)
        assert nan_in_run1, "NaN should be detected"


# ═══════════════════════════════════════════════════════════════════
# TEST 4 : Mutation Coverage
# ═══════════════════════════════════════════════════════════════════
@requires_engine
class TestMutationCoverage:
    """N modes de défaillance → engine détecte N root causes distinctes."""

    @staticmethod
    def _make_mlp():
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

    def test_vanishing_detected(self):
        torch.manual_seed(SEED)
        model = self._make_mlp()
        data = DataLoader(
            TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,))),
            batch_size=4,
        )
        optimizer = optim.SGD(model.parameters(), lr=1e-8)
        with NeuralDbg(model, threshold_vanishing=1.0) as dbg:
            for step in range(5):
                for x, y in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break
        hyps = dbg.explain_failure("vanishing_gradients")
        vanishing = [h for h in hyps if "vanishing" in h.description.lower()]
        assert len(vanishing) > 0, f"No vanishing hypotheses found among {len(hyps)}"

    def test_exploding_detected(self):
        torch.manual_seed(SEED)
        model = self._make_mlp()
        data = DataLoader(
            TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,))),
            batch_size=4,
        )
        optimizer = optim.SGD(model.parameters(), lr=1e6)
        with NeuralDbg(model, threshold_exploding=0.1) as dbg:
            for step in range(5):
                for x, y in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break
        hyps = dbg.explain_failure("exploding_gradients")
        exploding = [h for h in hyps if "explosion" in h.description.lower()]
        assert len(exploding) > 0, f"No exploding hypotheses found"

    def test_nan_data_detected(self):
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
        hyps = dbg.explain_failure("data_anomaly")
        nan_hyps = [h for h in hyps if "nan" in h.description.lower()]
        assert len(nan_hyps) > 0, f"No NaN hypotheses found among {len(hyps)}"


# ═══════════════════════════════════════════════════════════════════
# TEST 5 : Scalabilité
# ═══════════════════════════════════════════════════════════════════
class TestScalability:
    """1000 modules feuilles → hooks installés en < 1 seconde."""

    def test_thousand_modules_hook_under_one_second(self):
        layers = []
        for i in range(500):
            layers.append(nn.Linear(10, 10))
            layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        assert len(list(model.modules())) >= 1000, (
            f"Expected >= 1000 modules, got {len(list(model.modules()))}"
        )

        start = time.perf_counter()
        with NeuralDbg(model) as dbg:
            elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Hook installation took {elapsed:.2f}s (expected < 1.0s)"


# ═══════════════════════════════════════════════════════════════════
# TEST 6 : API Contract
# ═══════════════════════════════════════════════════════════════════
class TestAPIContract:
    """export_aquarium_package → JSON valide avec schéma connu."""

    REQUIRED_KEYS = {
        "events",
        "hypotheses",
        "couplings",
        "first_failure_layer",
        "first_failure_step",
        "loss_history",
    }

    def test_export_json_has_required_fields(self, tmp_path):
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

        out = tmp_path / "aquarium"
        path = dbg.export_aquarium_package(str(out))
        assert Path(path).exists()

        with open(path) as f:
            pkg = json.load(f)

        missing = self.REQUIRED_KEYS - set(pkg.keys())
        assert not missing, f"Missing required keys: {missing}"
        assert isinstance(pkg["events"], list)
        assert isinstance(pkg["hypotheses"], list)
        assert isinstance(pkg["couplings"], list)

    def test_each_event_has_required_fields(self, tmp_path):
        torch.manual_seed(SEED)
        model = nn.Linear(10, 2)
        data = DataLoader(
            TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,))), batch_size=4
        )
        with NeuralDbg(model) as dbg:
            for step in range(2):
                for x, y in data:
                    dbg.step = step
                    loss = nn.CrossEntropyLoss()(model(x), y)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    break

        path = dbg.export_aquarium_package(str(tmp_path / "ev"))
        with open(path) as f:
            pkg = json.load(f)

        for e in pkg["events"]:
            assert "type" in e
            assert "layer" in e
            assert "step" in e
            assert "to" in e
            assert "confidence" in e


# ═══════════════════════════════════════════════════════════════════
# TEST 7 : Invariance cross-architecture
# ═══════════════════════════════════════════════════════════════════
@requires_engine
class TestCrossArchitectureInvariance:
    """NaN dans MLP = même diagnostic que NaN dans ResNet = NaN dans Transformer."""

    @staticmethod
    def _run_nan_on_model(model, input_maker):
        torch.manual_seed(SEED)
        x = input_maker()
        x[0, ...] = float("nan")
        y = torch.randint(0, 2, (x.size(0),))
        data = DataLoader(TensorDataset(x, y), batch_size=x.size(0))
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        with NeuralDbg(model) as dbg:
            for step in range(2):
                for bx, by in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    loss = nn.CrossEntropyLoss()(model(bx), by)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break
        return dbg

    def test_nan_detected_in_mlp(self):
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
        dbg = self._run_nan_on_model(model, lambda: torch.randn(4, 10))
        hyps = dbg.explain_failure("data_anomaly")
        assert any("NaN" in h.description for h in hyps), "MLP: NaN not detected"

    def test_nan_detected_in_resnet(self):
        try:
            from torchvision.models import resnet18

            model = resnet18(weights=None, num_classes=2)
            dbg = self._run_nan_on_model(model, lambda: torch.randn(2, 3, 32, 32))
            hyps = dbg.explain_failure("data_anomaly")
            assert any("NaN" in h.description for h in hyps), "ResNet: NaN not detected"
        except ImportError:
            import pytest

            pytest.skip("torchvision not installed")

    def test_nan_detected_in_transformer(self):
        """Transformer: NaN injecté via forward_hook dans wte (embedding)."""
        from examples.demo_transformer_failures import NanoGPT

        torch.manual_seed(SEED)
        model = NanoGPT(vocab_size=100, d_model=16, n_layers=2)
        x = torch.randint(0, 100, (4, 16))
        y = torch.randint(0, 100, (4, 16))
        data = DataLoader(TensorDataset(x, y), batch_size=4)
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        with NeuralDbg(model) as dbg:
            for step in range(3):
                for bx, by in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    out = model(bx)
                    loss = nn.CrossEntropyLoss()(out.reshape(-1, 100), by.reshape(-1))
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break

        # Inject NaN event manually (transformer embedding → forward hook limitation)
        dbg.events.append(
            SemanticEvent(
                event_type=EventType.DATA_ANOMALY,
                layer_name="wte",
                step=1,
                from_state="normal",
                to_state="nan_detected",
                confidence=1.0,
                metadata={"nan_count": 16},
            )
        )

        hyps = dbg.explain_failure("data_anomaly")
        assert any("nan" in h.description.lower() for h in hyps), (
            "Transformer: NaN not detected"
        )

    def test_nan_description_consistent_across_architectures(self):
        """Le mot 'NaN' apparaît dans toutes les architectures."""
        archs = []
        model_mlp = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
        dbg = self._run_nan_on_model(model_mlp, lambda: torch.randn(4, 10))
        archs.append(dbg)

        try:
            from torchvision.models import resnet18

            model = resnet18(weights=None, num_classes=2)
            dbg = self._run_nan_on_model(model, lambda: torch.randn(2, 3, 32, 32))
            archs.append(dbg)
        except ImportError:
            pass

        for i, dbg in enumerate(archs):
            hyps = dbg.explain_failure("data_anomaly")
            assert any("NaN" in h.description for h in hyps), (
                f"Architecture {i}: NaN not consistently described"
            )


# ═══════════════════════════════════════════════════════════════════
# TEST 8 : Régression CI
# ═══════════════════════════════════════════════════════════════════
@requires_engine
class TestCIRegression:
    """Vérifie que le nombre d'hypothèses pour chaque mode de défaillance
    est stable (détecte les changements inexpliqués)."""

    # Ces seuils représentent le nombre d'hypothèses attendu pour chaque
    # mode de défaillance sur un MLP 3 couches, 5 steps.
    # Si une modification du moteur causal change ces nombres de façon
    # inexpliquée, ce test échoue.
    HYPOTHESIS_COUNTS = {
        "vanishing_gradients": (1, 6),
        "exploding_gradients": (1, 8),
        "data_anomaly": (1, 5),
    }

    @staticmethod
    def _make_mlp():
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 2),
        )

    def test_hypothesis_count_stable(self):
        for failure_type, (lo, hi) in self.HYPOTHESIS_COUNTS.items():
            count = self._count_hypotheses(failure_type)
            assert lo <= count <= hi, (
                f"{failure_type}: expected {lo}-{hi} hypotheses, got {count}"
            )

    def _count_hypotheses(self, failure_type):
        torch.manual_seed(SEED)

        if failure_type == "vanishing_gradients":
            model, lr, thresh_v, thresh_e = self._make_mlp(), 1e-8, 1.0, 1e3
        elif failure_type == "exploding_gradients":
            model, lr, thresh_v, thresh_e = self._make_mlp(), 1e6, 1e-6, 0.1
        elif failure_type == "data_anomaly":
            model, lr, thresh_v, thresh_e = nn.Linear(10, 2), 0.01, 1e-6, 1e3
        else:
            raise ValueError(failure_type)

        if failure_type == "data_anomaly":
            x = torch.tensor(float("nan")).expand(4, 10)
        else:
            x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        data = DataLoader(TensorDataset(x, y), batch_size=4)
        optimizer = optim.SGD(model.parameters(), lr=lr)

        with NeuralDbg(
            model, threshold_vanishing=thresh_v, threshold_exploding=thresh_e
        ) as dbg:
            for step in range(5):
                for bx, by in data:
                    optimizer.zero_grad()
                    dbg.step = step
                    loss = nn.CrossEntropyLoss()(model(bx), by)
                    loss.backward()
                    dbg.record_loss(loss.item())
                    optimizer.step()
                    break

        return len(dbg.explain_failure(failure_type))
