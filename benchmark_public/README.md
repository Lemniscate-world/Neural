# Public Causal Benchmark

Reproducible scoring of NeuralDBG on **synthetic** PyTorch failures with known ground truth.

## What it measures

| Metric | Meaning |
|--------|---------|
| **detection** | Hypothesis text mentions the expected failure mode |
| **localization** | Hypothesis references the correct layer id |
| **step_accuracy** | Hypothesis step within ±2 of injected failure step |
| **overall** | Mean of the three |

## Run locally

```bash
pip install -e ".[dev]"
python -m benchmark_public.run
```

Output: `benchmark_public/results.json` (committed for transparency).

## Why this helps

- Third parties can re-run the same script without trusting marketing copy.
- Regressions show up when `overall` drops on CI (optional gate).
- Complements dogfooding docs with a single number for README / papers.

**Note:** The full internal benchmark (7+ scenarios, engine-tuned thresholds) lives in the private engine repo and may score higher than this public subset.
