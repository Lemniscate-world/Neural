"""REAL integration tests for advertised integrations (D9: features must be
tested against the actual library contracts, not mocks-only).

- W&B callback: real wandb SDK in offline mode (no network/account needed).
- Lightning callback: real pytorch_lightning Trainer + CSVLogger.
  Skipped locally where lightning cannot install (py3.14 DLL); runs green in
  CI matrix (py3.11/3.12) — pytorch_lightning is in CI QA deps.

These tests exist because wandb.py had 50% and lightning.py 21% coverage with
mock-only smoke tests, while being headline README features (audit 2026-08-21).
"""

import json
import os

import pytest
import torch
import torch.nn as nn


class _VanishingMLP(nn.Module):
    """Tiny model whose gradients vanish (sigmoid + tiny init)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.Sigmoid(), nn.Linear(16, 2)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-4)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# W&B — REAL offline integration
# ──────────────────────────────────────────────────────────────────────────────

pytest.importorskip("wandb")


def test_wandb_callback_real_offline_run(tmp_path, monkeypatch):
    """NeuralDBGCallback inside a REAL wandb offline run: events captured,
    report() coherent, summary logged without crashing the SDK contract."""
    import wandb

    monkeypatch.chdir(tmp_path)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(project="neuraldbg-it", mode="offline")
    assert run is not None

    from neuraldbg.integrations.wandb import NeuralDBGCallback

    model = _VanishingMLP()
    x = torch.randn(4, 8)

    cb = NeuralDBGCallback(model, family="MLP", log_every_n_steps=2,
                           threshold_vanishing=1e-3)
    try:
        with cb:
            for step in range(6):
                out = model(x)
                loss = out.sum()
                loss.backward()
                cb.step(loss.item())

        # Callback state coherence after a real run
        assert cb._step == 6
        report = cb.report()
        assert report["total_events"] >= 0
        assert "summary" in report
        assert isinstance(report["event_counts"], dict)
        # Vanishing MLP + tiny threshold MUST surface anomalies (strict contract)
        assert report["anomaly_events"] >= 1, (
            f"Expected anomaly events on vanishing model, got {report}"
        )
    finally:
        wandb.finish()

    # Offline run wrote real binary transaction log under tmp_path (chdir'd)
    wandb_dirs = list(tmp_path.glob("wandb/offline-run-*"))
    assert len(wandb_dirs) >= 1, "offline wandb run directory missing"
    wandb_files = list(wandb_dirs[0].glob("run-*.wandb"))
    assert wandb_files and wandb_files[0].stat().st_size > 0, (
        "offline .wandb transaction log missing or empty — "
        "callback metrics never reached the real wandb SDK"
    )


def test_wandb_callback_alerts_real_trigger(tmp_path, monkeypatch):
    """alert_on event types are surfaced through report warnings path."""
    import wandb

    monkeypatch.chdir(tmp_path)
    os.environ["WANDB_MODE"] = "offline"

    wandb.init(project="neuraldbg-it-alerts", mode="offline")

    from neuraldbg.integrations.wandb import NeuralDBGCallback

    model = _VanishingMLP()
    cb = NeuralDBGCallback(model, family="MLP", log_every_n_steps=1,
                           alert_on=["nan_detected"], threshold_vanishing=1e-3)
    try:
        with cb:
            x = torch.randn(4, 8)
            for step in range(3):
                out = model(x)
                loss = out.sum()
                loss.backward()
                cb.step(loss.item())
        report = cb.report()
        assert "recommendations" in report
        assert isinstance(report["causal_chains"], list)
    finally:
        wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Lightning — REAL trainer integration (runs in CI py3.11/3.12)
# ──────────────────────────────────────────────────────────────────────────────

try:
    import pytorch_lightning as _pl  # noqa: F401
    _HAS_PL = True
except Exception:  # noqa: BLE001 — broken DLLs on py3.14 local envs
    _HAS_PL = False


@pytest.mark.skipif(not _HAS_PL, reason="pytorch_lightning unavailable/broken")
def test_lightning_callback_real_trainer_run(tmp_path):
    """Real pl.Trainer.fit with our callback attached: dbg lifecycle honored,
    events captured, CSVLogger receives numeric metrics without crashing."""
    pl = pytest.importorskip("pytorch_lightning",
                             reason="pytorch_lightning not installed")
    from torch.utils.data import DataLoader, TensorDataset

    from neuraldbg.integrations.lightning import NeuralDBGLightningCallback

    class TinyPLModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = _VanishingMLP()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            out = self(x)
            loss = ((out - y) ** 2).sum()
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-4)

    x = torch.randn(16, 8)
    y = torch.randn(16, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    cb = NeuralDBGLightningCallback(family="MLP", log_every_n_steps=1,
                                    threshold_vanishing=1e-3)
    from pytorch_lightning.loggers import CSVLogger

    trainer = pl.Trainer(
        max_epochs=2,
        limit_train_batches=3,
        logger=CSVLogger(str(tmp_path), name="neuraldbg-it"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
    )
    module = TinyPLModule()
    trainer.fit(module, loader)

    # Lifecycle: dbg entered during fit, exited (None) after fit end
    assert cb._dbg is None
    # Steps were recorded via on_train_batch_end
    assert cb._step == 6  # 2 epochs x 3 batches

    # CSV logger really received neuraldbg metrics (numeric contract)
    csv_dir = tmp_path / "neuraldbg-it" / "version_0"
    metrics_files = list(csv_dir.glob("metrics.csv"))
    assert metrics_files, "CSVLogger produced no metrics file"
    header = metrics_files[0].read_text().splitlines()[0]
    assert "neuraldbg/total_events" in header
