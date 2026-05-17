# HackerNews "Show HN" Draft

## Title
Show HN: NeuralDBG – Causal inference for PyTorch training dynamics

## URL
https://github.com/LambdaSection/NeuralDBG

## Text
I built NeuralDBG because I was tired of staring at TensorBoard curves guessing why my model failed. 

Most debugging tools show you *when* the loss spiked or vanished, but they don't tell you *why*. NeuralDBG analyzes gradients, activations, and data during training to provide structured causal hypotheses:

"Gradient vanishing originated in layer 'linear1' at step 234, likely due to LR × activation mismatch (confidence: 0.87)"

It's a Python package you wrap around your training loop. No dashboard setup, no cloud account, 100% local.

Key features:
- Semantic event extraction (detects transitions like Healthy → Vanishing)
- Post-mortem reasoning with ranked hypotheses
- Optimizer instability detection (plateaus, spikes, divergence)
- Data anomaly detection (NaN, Inf, distribution shifts)
- Works with torch.compile and distributed training

MIT License. Feedback welcome!
