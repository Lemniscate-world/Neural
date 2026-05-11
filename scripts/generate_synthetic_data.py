#!/usr/bin/env python3
"""Generate synthetic waveform datasets for DVC tracking and demo validation."""

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# Gradient norms over training steps (healthy → vanishing)
steps = np.arange(200)
gradient_norms = np.exp(-0.05 * steps) + rng.normal(0, 0.01, size=200)
np.save(DATA_DIR / "gradient_norms_demo.npy", gradient_norms)

# Activation distributions per layer over training steps
activations = rng.normal(0, 1, size=(200, 4))
activations[:, -1] *= np.exp(-0.08 * steps)  # last layer dies off
np.save(DATA_DIR / "activation_stats_demo.npy", activations)

print(f"Generated {len(list(DATA_DIR.glob('*.npy')))} files in {DATA_DIR}")
for f in sorted(DATA_DIR.glob("*.npy")):
    arr = np.load(f)
    print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}")
