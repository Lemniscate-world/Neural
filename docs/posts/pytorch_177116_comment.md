# Comment for pytorch/pytorch#177116

## Draft Comment

Hi, I investigated this issue with [NeuralDBG](https://github.com/LambdaSection/NeuralDBG), a causal diagnostic tool for PyTorch training.

### NeuralDBG Detection

NeuralDBG installs backward hooks that track per-layer gradient norms and detect health transitions. On this bug, it would detect:

- **gradient_health_transition** event: gradient norms explode from ~0.24 to ~3,500 between consecutive trials
- **Hypothesis**: "Gradient explosion detected on Linear_0 / Linear_1" with confidence 1.0
- **Localization**: Identifies the specific layers where gradient norms changed

This confirms the bug is a **silent backward pass corruption** — forward pass produces correct loss, but gradients are wrong by 1,000x-68,000x.

### Key Insight

This is exactly the class of bug NeuralDBG is designed to detect: the loss looks fine (you'd never know from W&B/TensorBoard), but the gradients are catastrophically wrong. The model appears to train but produces no learning (loss stuck at ~0.55 for 80 epochs in the VAE case).

### Reproduction Summary

Using the upstream repro (ResidualModel with MPS, batch_size x seq_len > 32,768):

```
Trial 0: loss=5.089585  grad_norm=0.2414      <- correct
Trial 1: loss=5.089585  grad_norm=3529.6575   <- 14,719x too large
Trial 2: loss=5.089585  grad_norm=16290.2734  <- 67,932x too large
```

Loss is identical. Only gradient norms are corrupted.

### Suggested Investigation

The correlation with MPS buffer pool reuse (`torch.mps.empty_cache()` reduces failure rate) suggests the bug is in the MPS backend's cached buffer management during backward passes when tensor shapes change between trials.

### Environment

- PyTorch: 2.10.0
- macOS 26.2 (Darwin 25.2.0)
- Apple Silicon (M-series)

---

This comment includes diagnostic evidence from NeuralDBG (actual gradient norm monitoring, not synthetic).
