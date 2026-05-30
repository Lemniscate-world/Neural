# PyTorch compatibility

| PyTorch | Python | Status |
|---------|--------|--------|
| 2.0.x | 3.11 | CI tested |
| 2.1.x | 3.11 | CI tested |
| 2.2.x | 3.11 | CI tested |
| 2.3.x | 3.11 | CI tested |
| 2.4.x | 3.12 | CI tested |
| 2.5.x | 3.12 | CI tested |
| 2.6.x | 3.12 | CI tested |
| 2.x + `torch.compile` | 3.10+ | Supported; some integration tests skip if compile unavailable |
| `nn.DataParallel` | 3.9+ | Supported (integration tests) |

## Install

```bash
pip install "neuraldbg>=1.3.1" "torch>=2.0"
```

## CI Matrix

The CI runs a dedicated `pytorch-compat` job that tests against 7 PyTorch versions (2.0.1 → 2.6.0) on CPU.
This ensures backward compatibility across the full supported range.

```yaml
# .github/workflows/ci.yml → pytorch-compat job
matrix:
  include:
    - pytorch-version: "2.0.1"  # python 3.11
    - pytorch-version: "2.1.2"  # python 3.11
    - pytorch-version: "2.2.2"  # python 3.11
    - pytorch-version: "2.3.1"  # python 3.11
    - pytorch-version: "2.4.1"  # python 3.12
    - pytorch-version: "2.5.1"  # python 3.12
    - pytorch-version: "2.6.0"  # python 3.12
```

## Known limits

- PyTorch only (no JAX/TF).
- `explain_failure()` quality is highest with optional `neuraldbg-engine`; core fallbacks cover common vanishing/exploding/data anomalies.
- Windows: use UTF-8 terminal or `quickstart.py` (no emoji) for CP1252 consoles.
