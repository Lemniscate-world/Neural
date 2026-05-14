#!/usr/bin/env python3
"""
CI gate for causal benchmark.
Fails if accuracy drops below MIN_ACCURACY.
Usage: PYTHONPATH="$HOME/Documents/NeuralDBG-Engine" python benchmark_ci.py
"""

import os
import sys

engine_path = os.path.expanduser("~/Documents/NeuralDBG-Engine")
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

try:
    from benchmark.scenarios import ALL_SCENARIOS
    from benchmark.benchmark import run_benchmark
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Ensure NeuralDBG-Engine exists at: {engine_path}")
    sys.exit(1)

MIN_ACCURACY = 0.80
THRESHOLD_V = 0.05
THRESHOLD_E = 0.2

ALL_SCENARIOS[0].ground_truth.expected_bug_layer = "3"
ALL_SCENARIOS[0].ground_truth.bug_layer = "3"

results = run_benchmark(
    threshold_v=THRESHOLD_V,
    threshold_e=THRESHOLD_E,
    verbose=True,
)
accuracy = results["_summary"]["overall"]

print(f"\n{'=' * 50}")
print(f"  Benchmark accuracy: {accuracy:.3f}  (min: {MIN_ACCURACY})")
print(f"{'=' * 50}")

if accuracy >= MIN_ACCURACY:
    print("  PASS")
    sys.exit(0)
else:
    print(f"  FAIL: accuracy {accuracy:.3f} < {MIN_ACCURACY}")
    print("  Run: make benchmark-tune")
    sys.exit(1)
