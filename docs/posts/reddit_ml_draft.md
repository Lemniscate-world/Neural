[D] I tested my PyTorch debugger against 212 architectures — here's what failed and why

Hey r/ML,

I built NeuralDBG, a causal diagnostic tool for PyTorch training failures. Instead of just showing you loss curves, it traces root causes through causal chains (gradient health → activation regime → optimizer instability → NaN).

To validate it, I tested against **212 architecture configurations** spanning 8 families: MLP, CNN, RNN, Transformer, Hybrid, GNN, Mixture of Experts, Diffusion, RL (Actor-Critic), and RAG. Each was tested with injected bugs (exploding LR, vanishing gradients, zero init, NaN data, dead bias, divergence).

## Results

| Tier | Architectures | Detection Rate |
|---|---|---|
| **Tier 1** | GNN, MoE, Diffusion | **96%** |
| **Tier 2** | FlashAttention, Neural ODE, Quantized | **94%** |
| **Tier 4** | RL Actor-Critic, RAG | Confirmed |
| RNN (LSTM/GRU) | — | 71% |
| Overall (212 configs) | — | 79% global |

Breakdown by family:
- MoE: 100%  
- Diffusion: 100%  
- FlashAttention: 100%  
- Neural ODE: 100%  
- GNN: 88%  
- Quantized (INT8/INT4): 83%  
- RNN: 71%  

## Real bugs (10 post-mortems)

Beyond synthetic tests, I used NeuralDBG to diagnose **10 real PyTorch / HuggingFace bugs**:

| Bug | Causal chain NeuralDBG traced | Upstream |
|---|---|---|
| `svdvals()` silently swallows NaN | `silent_corruption` — matrix rank = full on NaN | [PR #188053](https://github.com/pytorch/pytorch/pull/188053) |
| `F.normalize()` returns 0 instead of NaN at zero | `data_anomaly` — zero-vector guard | [PR #188066](https://github.com/pytorch/pytorch/pull/188066) |
| MPS gradients 100x-100Kx wrong | `gradient_health_transition` NORMAL→EXPLODING | [PR #188923](https://github.com/pytorch/pytorch/pull/188923) |
| `varlen_attn` silent NaN with padding | `gradient_health_transition` vanishing→NaN | [PR #188933](https://github.com/pytorch/pytorch/pull/188933) |
| Qwen3.5 SDPA explosion (HF#44928) | `sdpa_gradient_explosion` — dense mask → Math backend → BF16 | `attn_implementation=flash_attention_2` |
| Gradient clipping underflow | `optimizer_divergence` — grad norm below 1e-6 | Documented |
| AdamW + LayerNorm scale collapse | `activation_regime` → `gradient_health_transition` | Documented |
| FP16 softmax silent saturation | `silent_corruption` — logits exceed FP16 range | Documented |
| LSTM sample independence violation | `sample_independence_violation` (new event type) | [Comment posted](https://github.com/pytorch/pytorch/issues/173334) |
| torch.compile atan2 wrong grad | `gradient_health_transition` inductor vs eager | Catalogued |

4 PRs submitted upstream. 2 are marked MERGEABLE.

## Biggest surprises

1. **MoE went 0% → 100% from a hook bug, not a detection bug.** Hooks weren't reaching `ModuleList` children. Once fixed, 100% immediately. The lesson: hook attachment is where most "detection failures" actually live.

2. **RNNs are genuinely harder.** PyTorch's LSTM returns `(output, (h_n, c_n))` tuples, which silently breaks standard hook patterns. Added per-gate tracking (input/forget/cell/output gates separately). Still 71% — the remaining 29% are tuple-related edge cases.

3. **Family-aware thresholds matter more than model size.** A gradient norm of 10.0 is "exploding" in an MLP but completely normal in a GNN with 5 message-passing layers. One global threshold → 30% false positives.

4. **The causal chain `data_anomaly → gradient_explosion → optimizer_divergence` is real.** It appears in 6 out of 10 post-mortems with minor variations.

## What's next (Neural Suite)

The full pipeline is now end-to-end:
1. **NeuralDBG** detects and classifies the failure
2. Exports a structured JSON package (events + causal hypotheses)  
3. **Neural-Agent** reads the package and patches the config (`lr`, `clip_grad_norm`, `attn_implementation`, etc.)
4. **NeuralPrune** runs pruning diagnostics (dead neurons, low-rank layers, quantizable channels)

## Try it / Questions

- GitHub: https://github.com/LambdaSection/NeuralDBG (MIT)
- `pip install neuraldbg`
- Demo notebook: linked in README

How do you currently debug training failures? Print statements? W&B? Something custom? Curious what failure modes people hit in production that aren't in my catalog yet.
