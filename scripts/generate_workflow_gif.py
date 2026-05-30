#!/usr/bin/env python3
"""Generate an animated GIF showing how NeuralDBG works.

Creates a step-by-step walkthrough:
  1. Code integration (one-line wrapper)
  2. Semantic event extraction (what NeuralDBG sees)
  3. Causal reasoning (hypotheses generated)
  4. Final diagnosis

Usage:
    python scripts/generate_workflow_gif.py
Output:
    outputs/neuraldbg_workflow.gif
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import textwrap

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BG = "#0d1117"
FG = "#c9d1d9"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
ORANGE = "#db6d28"
DIM = "#484f58"
CARD_BG = "#161b22"


def _make_frame(func, *args, **kwargs):
    """Create a figure, call func, save to PNG, return path."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    func(ax, *args, **kwargs)
    path = OUTPUT_DIR / "_frames" / f"frame_{kwargs.get('frame_idx', 0):02d}.png"
    fig.savefig(path, dpi=120, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return path


def _draw_title(ax, text, y=6.6):
    ax.text(
        6,
        y,
        text,
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color=FG,
        fontfamily="sans-serif",
    )


def _draw_step_badge(ax, step_num, y=6.0):
    circle = plt.Circle((0.8, y), 0.3, color=ACCENT, zorder=5)
    ax.add_patch(circle)
    ax.text(
        0.8,
        y,
        str(step_num),
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=BG,
        zorder=6,
    )


def _draw_card(ax, x, y, w, h, title, lines, title_color=ACCENT):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.15",
        facecolor=CARD_BG,
        edgecolor=DIM,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.2,
        y + h - 0.25,
        title,
        fontsize=12,
        fontweight="bold",
        color=title_color,
        va="top",
        zorder=3,
    )
    for i, line in enumerate(lines):
        ax.text(
            x + 0.3,
            y + h - 0.6 - i * 0.35,
            line,
            fontsize=9.5,
            color=FG,
            va="top",
            fontfamily="monospace",
            zorder=3,
        )


def _draw_code_block(ax, x, y, w, h, code_lines):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.15",
        facecolor="#0d1117",
        edgecolor=DIM,
        linewidth=1,
        zorder=2,
    )
    ax.add_patch(rect)
    for i, (text, color) in enumerate(code_lines):
        ax.text(
            x + 0.25,
            y + h - 0.35 - i * 0.32,
            text,
            fontsize=8.5,
            color=color,
            va="top",
            fontfamily="monospace",
            zorder=3,
        )


def _draw_arrow(ax, x1, y1, x2, y2, color=ACCENT):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
        zorder=4,
    )


# ── Frame 0: Title ──────────────────────────────────────────────
def _frame_title(ax, frame_idx=0):
    ax.text(
        6,
        5.5,
        "NeuralDBG",
        ha="center",
        va="center",
        fontsize=42,
        fontweight="bold",
        color=ACCENT,
        fontfamily="sans-serif",
    )
    ax.text(
        6,
        4.7,
        "Causal Inference Engine for Deep Learning",
        ha="center",
        va="center",
        fontsize=16,
        color=FG,
    )
    ax.text(
        6,
        4.0,
        "TensorBoard tells you WHEN it failed.\nNeuralDBG tells you WHY.",
        ha="center",
        va="center",
        fontsize=14,
        color=YELLOW,
        style="italic",
    )
    ax.text(
        6,
        2.8,
        "3 scenarios  ·  100% detection  ·  100% localization",
        ha="center",
        va="center",
        fontsize=12,
        color=GREEN,
    )
    ax.text(
        6,
        1.5,
        "How it works ↓",
        ha="center",
        va="center",
        fontsize=14,
        color=ACCENT,
        fontweight="bold",
    )


# ── Frame 1: Code Integration ───────────────────────────────────
def _frame_code(ax, frame_idx=1):
    _draw_title(ax, "Step 1: One-Line Integration")
    _draw_step_badge(ax, 1)

    code = [
        ("import torch", FG),
        ("from neuraldbg import NeuralDbg", FG),
        ("", FG),
        ("model = MyModel()", FG),
        ("", FG),
        ("# Wrap your training loop", DIM),
        ("with NeuralDbg(model) as dbg:", GREEN),
        ("    for step, (x, y) in dataloader:", FG),
        ("        loss = train_step(model, x, y)", FG),
        ("        dbg.record_loss(loss.item())", ACCENT),
        ("", FG),
        ("# After failure — query explanations", DIM),
        ("hypotheses = dbg.explain_failure()", GREEN),
    ]
    _draw_code_block(ax, 0.5, 0.4, 5.5, 5.5, code)

    _draw_card(
        ax,
        6.8,
        3.5,
        4.8,
        2.5,
        "What NeuralDBG captures",
        [
            "· Gradient norms per layer",
            "· Activation statistics",
            "· Loss trajectory",
            "· Data distribution shifts",
            "· Memory usage",
        ],
        GREEN,
    )

    _draw_card(
        ax,
        6.8,
        0.5,
        4.8,
        2.5,
        "Zero config needed",
        [
            "· No new API to learn",
            "· Works with any PyTorch model",
            "· Survives torch.compile",
            "· 100% local (no cloud)",
        ],
        ACCENT,
    )

    _draw_arrow(ax, 6.0, 3.2, 6.8, 3.2, ACCENT)


# ── Frame 2: Event Detection ────────────────────────────────────
def _frame_events(ax, frame_idx=2):
    _draw_title(ax, "Step 2: Semantic Event Extraction")
    _draw_step_badge(ax, 2)

    events = [
        (
            "gradient_health_transition",
            "Linear_0",
            "step 2",
            "healthy → vanishing",
            RED,
        ),
        ("activation_regime_shift", "ReLU_1", "step 2", "normal → saturated", ORANGE),
        ("data_anomaly", "root", "step 0", "normal → distribution_shift", YELLOW),
        (
            "gradient_health_transition",
            "Linear_2",
            "step 2",
            "healthy → vanishing",
            RED,
        ),
        ("optimizer_instability", "root", "step 5", "stable → loss_spike", ORANGE),
    ]

    y_start = 5.5
    for i, (etype, layer, step, transition, color) in enumerate(events):
        y = y_start - i * 0.95
        # Event type badge
        rect = mpatches.FancyBboxPatch(
            (0.5, y - 0.25),
            3.8,
            0.6,
            boxstyle="round,pad=0.1",
            facecolor=CARD_BG,
            edgecolor=color,
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            0.7,
            y,
            etype.replace("_", " "),
            fontsize=8,
            color=color,
            va="center",
            fontfamily="monospace",
            zorder=3,
        )

        # Layer + step
        ax.text(
            4.6,
            y,
            f"{layer}  {step}",
            fontsize=9,
            color=FG,
            va="center",
            fontfamily="monospace",
            zorder=3,
        )

        # Transition arrow
        ax.text(
            6.8,
            y,
            transition,
            fontsize=9,
            color=color,
            va="center",
            fontweight="bold",
            zorder=3,
        )

    _draw_card(
        ax,
        7.5,
        1.0,
        4.2,
        4.5,
        "Key insight",
        [
            "NeuralDBG does NOT track",
            "raw tensors.",
            "",
            "It extracts semantic events:",
            "  · transitions, not values",
            "  · first occurrences",
            "  · propagation patterns",
            "",
            "→ Compact, causal, actionable",
        ],
        YELLOW,
    )

    _draw_arrow(ax, 6.8, 3.5, 7.5, 3.5, ACCENT)


# ── Frame 3: Causal Hypotheses ──────────────────────────────────
def _frame_hypotheses(ax, frame_idx=3):
    _draw_title(ax, "Step 3: Causal Hypothesis Generation")
    _draw_step_badge(ax, 3)

    # Hypothesis cards
    hyps = [
        {
            "rank": "Rank 1",
            "desc": "Gradient vanishing originated in\nlayer 'ReLU_1' at step 2",
            "confidence": "1.00",
            "chain": "ReLU_1@2 → Linear_0@2 → root@2",
            "color": RED,
        },
        {
            "rank": "Rank 2",
            "desc": "Root cause: data distribution shift\noriginated in 'root' at step 0",
            "confidence": "0.95",
            "chain": "root@0 → Linear_0@0 → ReLU_1@0",
            "color": YELLOW,
        },
        {
            "rank": "Rank 3",
            "desc": "Optimizer instability detected\nat root (step 5)",
            "confidence": "0.80",
            "chain": "root@5 (loss spike)",
            "color": ORANGE,
        },
    ]

    for i, hyp in enumerate(hyps):
        x = 0.5 + i * 3.9
        rect = mpatches.FancyBboxPatch(
            (x, 0.8),
            3.6,
            4.8,
            boxstyle="round,pad=0.2",
            facecolor=CARD_BG,
            edgecolor=hyp["color"],
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.3,
            5.2,
            hyp["rank"],
            fontsize=11,
            fontweight="bold",
            color=hyp["color"],
            va="top",
            zorder=3,
        )
        ax.text(
            x + 0.3,
            4.5,
            hyp["desc"],
            fontsize=9,
            color=FG,
            va="top",
            fontfamily="monospace",
            zorder=3,
            linespacing=1.4,
        )
        ax.text(x + 0.3, 3.0, "Confidence:", fontsize=9, color=DIM, va="top", zorder=3)
        ax.text(
            x + 0.3,
            2.5,
            hyp["confidence"],
            fontsize=20,
            fontweight="bold",
            color=GREEN,
            va="top",
            zorder=3,
        )
        ax.text(
            x + 0.3, 1.8, "Causal chain:", fontsize=9, color=DIM, va="top", zorder=3
        )
        ax.text(
            x + 0.3,
            1.3,
            hyp["chain"],
            fontsize=8,
            color=ACCENT,
            va="top",
            fontfamily="monospace",
            zorder=3,
        )

    _draw_arrow(ax, 6, 5.8, 6, 5.6, ACCENT)


# ── Frame 4: Summary / Why it matters ───────────────────────────
def _frame_summary(ax, frame_idx=4):
    _draw_title(ax, "Why NeuralDBG?")

    comparisons = [
        (
            "TensorBoard / W&B",
            [
                "Shows loss/accuracy curves",
                "Manual inspection required",
                "You guess the fix",
                "Separate dashboard",
                "Data sent to cloud",
            ],
            DIM,
        ),
        (
            "NeuralDBG",
            [
                "WHY it failed (causal)",
                "Automated hypotheses",
                "Suggests root causes",
                "One line of code",
                "100% local",
            ],
            GREEN,
        ),
    ]

    for col, (title, items, color) in enumerate(comparisons):
        x = 0.8 + col * 6.0
        rect = mpatches.FancyBboxPatch(
            (x, 1.5),
            5.2,
            4.5,
            boxstyle="round,pad=0.2",
            facecolor=CARD_BG,
            edgecolor=color,
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + 2.6,
            5.6,
            title,
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=color,
            va="top",
            zorder=3,
        )
        for i, item in enumerate(items):
            check = "✗" if col == 0 else "✓"
            ch_color = RED if col == 0 else GREEN
            ax.text(
                x + 0.4,
                4.8 - i * 0.6,
                f"{check}  {item}",
                fontsize=10,
                color=FG,
                va="center",
                zorder=3,
            )
            ax.text(
                x + 0.5,
                4.8 - i * 0.6,
                check,
                fontsize=10,
                color=ch_color,
                va="center",
                fontweight="bold",
                zorder=4,
            )

    # Quote
    ax.text(
        6,
        0.8,
        '"TensorBoard tells you when it failed.\nNeuralDBG tells you why."',
        ha="center",
        fontsize=13,
        color=YELLOW,
        style="italic",
        va="center",
    )

    # Benchmark badge
    rect = mpatches.FancyBboxPatch(
        (8.5, 6.0),
        3.2,
        0.7,
        boxstyle="round,pad=0.15",
        facecolor=GREEN,
        edgecolor=GREEN,
        linewidth=1,
        zorder=2,
        alpha=0.15,
    )
    ax.add_patch(rect)
    ax.text(
        10.1,
        6.35,
        "Benchmark: 100% accuracy",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=GREEN,
        va="center",
        zorder=3,
    )


def generate_workflow_gif():
    frames_dir = OUTPUT_DIR / "_frames"
    frames_dir.mkdir(exist_ok=True)

    print("Rendering frames...")
    frames_info = [
        (_frame_title, "NeuralDBG Overview"),
        (_frame_code, "Code Integration"),
        (_frame_events, "Event Extraction"),
        (_frame_hypotheses, "Causal Hypotheses"),
        (_frame_summary, "Why NeuralDBG"),
    ]

    frame_paths = []
    for i, (func, label) in enumerate(frames_info):
        path = _make_frame(func, frame_idx=i)
        frame_paths.append(path)
        print(f"  Frame {i + 1}: {label}")

    # Assemble GIF
    frames = [Image.open(p).convert("RGB") for p in frame_paths]

    gif_path = OUTPUT_DIR / "neuraldbg_workflow.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=2500,
        loop=0,
        optimize=False,
    )
    print(f"\nGIF: {gif_path} ({gif_path.stat().st_size / 1024:.1f} KB)")

    # Save last frame as static preview
    preview_path = OUTPUT_DIR / "neuraldbg_workflow_preview.png"
    frames[-1].save(preview_path)
    print(f"Preview: {preview_path}")

    # Cleanup
    for f in frames_dir.glob("*.png"):
        f.unlink()
    frames_dir.rmdir()

    return gif_path


if __name__ == "__main__":
    generate_workflow_gif()
