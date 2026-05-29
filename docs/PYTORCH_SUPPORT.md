# PyTorch compatibility

| PyTorch | Python | Status |
|---------|--------|--------|
| 2.0.x – 2.6.x | 3.9 – 3.12 | Supported (CI) |
| 2.x + `torch.compile` | 3.10+ | Supported; some integration tests skip if compile unavailable |
| `nn.DataParallel` | 3.9+ | Supported (integration tests) |

## Install

```bash
pip install "neuraldbg>=1.3.1" "torch>=2.0"
```

## Known limits

- PyTorch only (no JAX/TF).
- `explain_failure()` quality is highest with optional `neuraldbg-engine`; core fallbacks cover common vanishing/exploding/data anomalies.
- Windows: use UTF-8 terminal or `quickstart.py` (no emoji) for CP1252 consoles.
