[D] I tested my PyTorch debugger against 212 architectures — here's what failed and why

Hey r/ML,

I built NeuralDBG, a causal diagnostic tool for PyTorch training failures. Instead of just showing you loss curves, it traces root causes through causal chains (gradient health → activation regime → optimizer instability → NaN).

To validate it, I tested against **212 architecture configurations** spanning 8 families: MLP, CNN, RNN, Transformer, Hybrid, GNN, Mixture of Experts, and Diffusion models. Each was tested with 6 injected bugs (exploding LR, vanishing gradients, zero init, NaN data, dead bias, divergence).

## Results that surprised me

| Test | Detection Rate |
|------|:------------:|
| **GNN (Graph Neural Networks)** | 88% |
| **MoE (Mixture of Experts)** | 100% |
| **Diffusion models** | 100% |
| **FlashAttention** | 100% |
| **Neural ODE** | 100% |
| **Quantized (INT8/INT4)** | 83% |
| **RNN (LSTM/GRU)** | 71% |
| **Overall (Tier 1+2)** | **96% / 94%** |

The biggest surprises:
1. **MoE was easier than expected** — once hooks properly attached to ModuleList children, detection hit 100%. The hard part wasn't detection, it was the data pipeline (we had a shape mismatch bug that took hours to find).
2. **RNNs are genuinely harder** — PyTorch's LSTM/GRU return tuples `(output, (h_n, c_n))`, which silently breaks standard hook patterns. We had to add per-gate gradient tracking (input/forget/cell/output) to catch vanishing in specific gates.
3. **Quantized models create noise** — INT4 quantization introduces precision loss that masks some bug signals. Real quantization is harder to debug than simulated quantization.

## Real bugs we found along the way

We used NeuralDBG to diagnose 7 real PyTorch bugs, submitted 4 PRs:
- `torch.linalg.svdvals()` silently swallowing NaN (#187759)
- `F.normalize()` returning 0 instead of NaN at zero input (#184575)
- MPS gradient corruption producing 100x-100,000x wrong gradients (#177116)
- `varlen_attn()` producing silent NaN with padding tokens (#176793)

## What I learned about training failures

1. **Failures are local, not global** — the loss spike is the symptom, not the cause. The root cause is almost always a specific layer at a specific step.
2. **Causal chains work** — `data_anomaly → gradient_explosion → optimizer_divergence` is a real pattern that repeats across architectures.
3. **Family matters** — MLP and Transformer have different "noise floors" for gradient norms. A threshold that works for MLP gives false positives on CNN.

## Try it / Questions

- GitHub: https://github.com/LambdaSection/NeuralDBG (MIT)
- `pip install neuraldbg`

How do you currently debug training failures? Print statements? W&B? Something custom? Curious what works for people in production.
