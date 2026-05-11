# ROADMAP.md -- Vision

## Vision
NeuralDBG aims to be the standard causal debugging engine for deep learning, providing human-interpretable explanations for complex training failures.

## v1.2.x -- Stabilization (Current)
- Hardened governance and cross-platform infrastructure.
*   **Target**: Extreme Rigor in explanation quality for vanishing gradients.

## v1.3.0 -- Advanced Failure Families
- Support for complex architectural failures (e.g., Attention Collapse, ResNet Bottleneck Saturation).
- Integration with Weight & Biases / MLflow for real-time causal dashboarding.

## v1.5.0 -- Multi-Framework Support
- First-class support for JAX/Flax.
- Prototype support for TensorFlow.

## v2.0.0 -- Autonomous Remediation
- The engine not only explains failures but *proposes and applies* architectural or hyperparameter fixes (e.g., suggesting a different weight init or learning rate scheduler).

---
**Last Updated**: 2026-05-11
