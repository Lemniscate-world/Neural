import pytest
import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
    from neuraldbg.integrations.lightning import NeuralDBGLightningCallback

    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False


class LinearModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(16, 2)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.mark.skipif(not HAS_LIGHTNING, reason="pytorch_lightning not installed")
def test_lightning_callback_initializes():
    model = LinearModel()
    callback = NeuralDBGLightningCallback(family="MLP", log_every_n_steps=10)
    assert callback.family == "MLP"
    assert callback.log_every_n_steps == 10


@pytest.mark.skipif(not HAS_LIGHTNING, reason="pytorch_lightning not installed")
def test_lightning_callback_in_trainer():
    model = LinearModel()
    callback = NeuralDBGLightningCallback(family="MLP", log_every_n_steps=10)
    trainer = pl.Trainer(
        callbacks=[callback],
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    x = torch.randn(32, 16)
    y = torch.randint(0, 2, (32,))
    train_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=8)
    trainer.fit(model, train_loader)
    assert callback._step > 0


@pytest.mark.skipif(not HAS_LIGHTNING, reason="pytorch_lightning not installed")
def test_lightning_callback_events_logged():
    model = LinearModel()
    callback = NeuralDBGLightningCallback(family="MLP", log_every_n_steps=5)
    trainer = pl.Trainer(
        callbacks=[callback],
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    x = torch.randn(16, 16)
    y = torch.randint(0, 2, (16,))
    train_loader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=4)
    trainer.fit(model, train_loader)
    assert callback._dbg is not None
    events = callback._dbg.dump_events()
    assert isinstance(events, list)
