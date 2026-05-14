#!/usr/bin/env python3
"""
DDPM (Denoising Diffusion) failure scenarios demonstrating NeuralDBG.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from neuraldbg import NeuralDbg


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MiniUNet(nn.Module):
    """Minimal UNet for diffusion on synthetic 2D data (no spatial down/upsampling)."""

    def __init__(self, in_channels=1, base_dim=16):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_dim)
        self.enc2 = ConvBlock(base_dim, base_dim * 2)
        self.mid = ConvBlock(base_dim * 2, base_dim * 2)
        self.dec2 = ConvBlock(base_dim * 4, base_dim * 2)
        self.dec1 = ConvBlock(base_dim * 3, in_channels)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.avg_pool2d(e1, 2))
        e2_up = nn.functional.interpolate(e2, scale_factor=2)
        m = self.mid(e2)
        m_up = nn.functional.interpolate(m, scale_factor=2)
        d2 = self.dec2(torch.cat([m_up, e2_up], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1


class NoiseScheduler:
    def __init__(self, steps=100, beta_start=1e-4, beta_end=0.02):
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, step):
        noise = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[step]
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise, noise


def make_loader(batch_size=4, channels=1, size=8, num_samples=50):
    x = torch.randn(num_samples, channels, size, size)
    y = torch.zeros(num_samples, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def train_unet(
    model, dataloader, scheduler, num_steps=20, lr=0.001, nan_step=None, noise_scale=1.0
):
    """Train UNet with NeuralDBG monitoring."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    nan_injected = False

    with NeuralDbg(model, threshold_vanishing=1e-4, threshold_exploding=1.0) as dbg:
        for step in range(num_steps):
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                dbg.step = step

                t = torch.randint(0, min(scheduler.betas.shape[0], 10), (1,)).item()
                noisy, noise_target = scheduler.add_noise(batch_x, t)
                noisy = noisy * noise_scale

                if nan_step is not None and step >= nan_step and not nan_injected:
                    model.enc1.net[0].weight.data[0, 0, 0, 0] = float("nan")
                    nan_injected = True

                pred = model(noisy)
                loss = criterion(pred, noise_target)
                loss.backward()
                dbg.record_loss(loss.item())
                optimizer.step()
                break
    return dbg


def analyze_results(dbg):
    return {
        "hypotheses": dbg.explain_failure("vanishing_gradients")
        + dbg.explain_failure("exploding_gradients"),
        "opt_hypotheses": dbg.explain_failure("optimizer_instability"),
        "data_hypotheses": dbg.explain_failure("data_anomaly"),
        "couplings": dbg.detect_coupled_failures(),
        "events": dbg.events,
        "mermaid": dbg.export_mermaid_causal_graph(),
    }


def scenario_unet_nan(num_steps=20):
    """NaN injection in UNet encoder -> data anomaly."""
    model = MiniUNet()
    scheduler = NoiseScheduler(steps=50)
    loader = make_loader()
    return train_unet(
        model, loader, scheduler, num_steps=num_steps, lr=0.001, nan_step=5
    )


def scenario_unet_exploding(num_steps=20):
    """Very high LR -> gradients explode in deep UNet."""
    model = MiniUNet()
    scheduler = NoiseScheduler(steps=50)
    loader = make_loader()
    return train_unet(model, loader, scheduler, num_steps=num_steps, lr=100.0)


def scenario_unet_collapse(num_steps=20):
    """Noise scale too high -> signal overwhelmed -> model can't learn."""
    model = MiniUNet()
    scheduler = NoiseScheduler(steps=10, beta_start=0.1, beta_end=0.9)
    loader = make_loader()
    return train_unet(
        model, loader, scheduler, num_steps=num_steps, lr=0.001, noise_scale=100.0
    )


def main():
    torch.manual_seed(42)
    print("[NeuralDBG] DDPM UNet failure scenarios\n")

    for name, fn in [
        ("NaN in UNet encoder", scenario_unet_nan),
        ("Exploding gradients (LR=100)", scenario_unet_exploding),
        ("Noise schedule collapse (aggressive noise)", scenario_unet_collapse),
    ]:
        dbg = fn(num_steps=12)
        results = analyze_results(dbg)
        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {name}")
        print(f"{'=' * 60}")
        print(f"Events: {len(results['events'])}")
        for label, hyps in [
            ("Gradient hypotheses", results["hypotheses"]),
            ("Optimizer hypotheses", results["opt_hypotheses"]),
            ("Data anomaly", results["data_hypotheses"]),
        ]:
            if hyps:
                print(f"{label}:")
                for h in hyps[:3]:
                    print(f"  [{h.confidence:.2f}] {h.description}")

    print("\n[DONE] DDPM scenarios complete.")


if __name__ == "__main__":
    main()
