#!/usr/bin/env python3
"""Generate an animated GIF showing NeuralDBG vanishing gradient detection.

Usage:
    python scripts/generate_vanishing_gif.py
Output:
    outputs/vanishing_gradient_demo.gif
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import io

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _build_model():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.Tanh(),
        nn.Linear(20, 20),
        nn.Tanh(),
        nn.Linear(20, 2),
    )


def _collect_data(num_steps: int = 10, seed: int = 42):
    torch.manual_seed(seed)
    model = _build_model()
    x = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Collect per-parameter-layer gradient norms (skip activation-only layers)
    param_layer_names = []
    param_modules = []  # (name, module) pairs with parameters
    for name, mod in model.named_modules():
        if len(list(mod.children())) > 0 and name != "":
            continue
        if not any(True for _ in mod.parameters()):
            continue  # skip activation-only layers (Tanh, ReLU, etc.)
        if name == "":
            param_layer_names.append("root")
        else:
            param_layer_names.append(f"{type(mod).__name__}_{name}")
        param_modules.append((name, mod))

    norms_per_step = []
    for step in range(num_steps):
        optimizer.zero_grad()
        if step == 2:
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(0.0)
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()

        step_norms = []
        for name, mod in param_modules:
            for p in mod.parameters():
                if p.grad is not None:
                    step_norms.append(p.grad.norm().item())
                    break
            else:
                step_norms.append(1e-10)
        norms_per_step.append(step_norms)
        optimizer.step()

    return param_layer_names, norms_per_step


def _save_frame(
    layer_names: list[str],
    norms_matrix: np.ndarray,
    step: int,
    total_steps: int,
    filepath: Path,
):
    """Render and save a single frame to a file."""
    n_layers = len(layer_names)
    data = norms_matrix[: step + 1, :].T
    data = np.clip(data, 1e-7, None)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "health", ["#e74c3c", "#f39c12", "#2ecc71"], N=256
    )
    norm = mcolors.LogNorm(vmin=1e-6, vmax=1e-1)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[-0.5, total_steps - 0.5, n_layers - 0.5, -0.5],
    )

    # Injection line
    ax.axvline(x=1.5, color="#ff4444", linestyle="--", alpha=0.8, linewidth=2)
    ax.text(
        1.5,
        -0.7,
        "INJECT\n(step 2)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#ff4444",
        fontweight="bold",
    )

    # Mark vanishing cells with X
    for t in range(step + 1):
        for l in range(n_layers):
            if norms_matrix[t, l] < 1e-4:
                ax.plot(t, l, "x", color="white", markersize=10, markeredgewidth=2.5)

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_names, fontsize=9, fontfamily="monospace", color="white")
    ax.set_xticks(range(total_steps))
    ax.set_xticklabels(range(total_steps), color="white")
    ax.set_xlabel("Training Step", fontsize=11, color="white", labelpad=10)
    ax.set_ylabel("Layer", fontsize=11, color="white", labelpad=10)

    title = f"NeuralDBG — Vanishing Gradient Detection\nStep {step}/{total_steps - 1}"
    if step >= 2:
        n_van = np.sum(norms_matrix[step, :] < 1e-4)
        title += f" | {n_van}/{n_layers} layers vanishing"
    ax.set_title(title, fontsize=13, fontweight="bold", color="white", pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Gradient Norm (log)", fontsize=9, color="white")
    cbar.ax.tick_params(colors="white")

    # Legend
    legend = "X = vanishing (norm < 1e-4)"
    ax.text(
        0.02,
        0.98,
        legend,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#222", edgecolor="#555"),
    )

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout()
    fig.savefig(filepath, dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def generate_gif():
    print("Collecting gradient norms...")
    layer_names, norms_list = _collect_data()
    num_steps = len(norms_list)
    norms_matrix = np.array(norms_list)

    print(f"Layers: {layer_names}")
    print(f"Step 0: {norms_matrix[0]}")
    print(f"Step 2: {norms_matrix[2]}")

    # Save individual frames
    frame_dir = OUTPUT_DIR / "_frames"
    frame_dir.mkdir(exist_ok=True)

    print(f"Rendering {num_steps} frames...")
    for step in range(num_steps):
        frame_path = frame_dir / f"frame_{step:02d}.png"
        _save_frame(layer_names, norms_matrix, step, num_steps, frame_path)
        print(f"  Frame {step + 1}/{num_steps}")

    # Assemble GIF
    frames = []
    for step in range(num_steps):
        frame_path = frame_dir / f"frame_{step:02d}.png"
        img = Image.open(frame_path).convert("RGB")
        frames.append(img)

    gif_path = OUTPUT_DIR / "vanishing_gradient_demo.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=700,
        loop=0,
        optimize=False,
    )
    print(f"\nGIF: {gif_path} ({gif_path.stat().st_size / 1024:.1f} KB)")

    # Final frame
    png_path = OUTPUT_DIR / "vanishing_gradient_final.png"
    frames[-1].save(png_path)
    print(f"PNG: {png_path}")

    # Cleanup frames
    for f in frame_dir.glob("*.png"):
        f.unlink()
    frame_dir.rmdir()

    return gif_path


if __name__ == "__main__":
    generate_gif()
