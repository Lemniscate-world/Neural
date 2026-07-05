# Black-Swan Architecture Catalog — NeuralSuite

> "Expect the worst. Detect the non-existent. +10 resilience on every module."
> Strategy updated 2026-07-05.

---

## Tested (200+ configs, combinatorial sweep)

| Family | Variants Tested | Detection |
|--------|:--------------:|:---------:|
| MLP | 50 (depth 2-10, width 16-256, ReLU/GELU/SiLU/Tanh/LeakyReLU/ELU, BatchNorm/LayerNorm/None, skip/no-skip) | 93-98% |
| CNN | 40 (Conv2d, depth 2-5, kernel 3/5, ReLU/GELU/LeakyReLU, BatchNorm/None, skip) | 90-93% |
| RNN | 40 (LSTM/GRU, depth 1-4, bidirectional, width 32-256) | 65-70% |
| Transformer | 40 (MHA, depth 1-4, heads 2-8, d_model 32-128, GELU/ReLU, LayerNorm) | 91-93% |
| Hybrid | 30 (conv+mlp, attn+mlp, rnn+mlp, cnn+rnn+mlp, all combined) | 36-96% |
| Paper archs | 60 (Mamba, KAN, xLSTM, MoE, Hyena, RWKV, BitNet, RetNet, Griffin, Jamba, S4, NeuralODE, LTC, etc.) | Not yet tested |

---

## Black-Swan Architectures — NOT YET TESTED

### A. Architectural Black Swans (unknown failure modes)

| # | Architecture | Why Black Swan | Risk |
|---|-------------|----------------|------|
| 1 | **Graph Neural Networks (GCN/GAT/GIN)** | Message passing has different gradient flow than feed-forward. Aggregation ops (scatter_add) can silently overflow. | HIGH |
| 2 | **FlashAttention / PagedAttention** | Custom CUDA kernels bypass PyTorch autograd hooks. Silent NaN, wrong gradients. | CRITICAL |
| 3 | **Mixture of Experts (MoE) with load balancing** | Auxiliary loss interacts with main loss. Gradient interference between experts. Dead expert collapse. | HIGH |
| 4 | **Diffusion Models / UNet** | Timestep conditioning creates unique gradient paths. Skip connections across vast depth. | HIGH |
| 5 | **Neural ODEs / Continuous-depth** | Adjoint method for gradients. Different numerical stability profile than discrete layers. | MEDIUM |
| 6 | **Quantized models (INT8/INT4, GPTQ, AWQ)** | Fake quantization nodes. Gradient scaling factors can underflow. | HIGH |
| 7 | **Multi-modal (CLIP, LLaVA, Flamingo)** | Contrastive loss + cross-modal gradients. Modality-specific vanishing. | MEDIUM |
| 8 | **Retrieval-Augmented (RAG)** | Non-differentiable retrieval step breaks gradient chain. Hybrid discrete/continuous. | MEDIUM |
| 9 | **Reinforcement Learning (Actor-Critic, DQN)** | Non-stationary targets. TD-error explosion. Policy gradient variance. | HIGH |
| 10 | **Federated Learning** | Model aggregation across heterogeneous clients. Divergent gradient statistics. | HIGH |

### B. Hardware Black Swans

| # | Failure Mode | Example |
|---|-------------|--------|
| 1 | **MPS backend gradient corruption** | BUG-003: gradients 100-100,000x too large on Apple Silicon |
| 2 | **XLA/TPU compilation quirks** | torch.compile can reorder ops, changing numerical precision |
| 3 | **Mixed precision (fp16/bf16) underflow** | Gradient underflow in fp16, undetected by fp32 checks |
| 4 | **Distributed training race conditions** | AllReduce ordering, gradient staleness in async training |
| 5 | **CUDA graph replay inconsistencies** | Cached allocations, stale gradient buffers |

### C. Numerical Black Swans

| # | Failure Mode | Risk |
|---|-------------|------|
| 1 | **Catastrophic cancellation in LayerNorm** | x - E[x] can lose precision for large values |
| 2 | **Softmax saturation in long sequences** | 100K+ token attention: exp overflow |
| 3 | **Log-space instability (log_softmax + NLL)** | Underflow in log domain |
| 4 | **Einsum / scatter_reduce silent corruption** | Undefined behavior for duplicate indices |
| 5 | **SVD / eigenvalue gradient instability** | Near-degenerate matrices cause gradient explosion |

### D. Interaction Black Swans

| # | Combination | Risk |
|---|------------|------|
| 1 | **Gradient clipping + LayerNorm** | Clipping before norm = different effective LR per layer |
| 2 | **Weight decay + AdamW + fp16** | Decay in fp16 can underflow to zero |
| 3 | **Dropout + BatchNorm interaction** | Variance shift between train/eval |
| 4 | **EMA + SWA + mixed precision** | Shadow weights in different precision |
| 5 | **torch.compile + custom autograd.Function** | Dynamo guards can miss custom backward |

---

## Black-Swan Detection Strategy

### Tier 1: Known Unknowns (test this week)
- Add GNN, MoE, Diffusion model families to combinatorial tester
- Add FlashAttention hook bypass detection
- Add mixed-precision gradient health checks

### Tier 2: Unknown Unknowns (infrastructure)
- **Fuzzer**: random architecture generator that mutates valid models
- **Adversarial bug injector**: gradient corruption at random layers
- **Stress test**: extreme values (1e-10 to 1e+10 inputs, weights, grads)
- **Hardware matrix**: test on CPU/GPU/MPS/TPU with same seeds

### Tier 3: Predictive (research)
- Train anomaly detector on normal training dynamics
- Flag ANY deviation from learned normal patterns
- Zero-config: no need to pre-define failure modes

---

## +10 Resilience Target

For every module/detector in NeuralDBG, target ability to handle:
- 10x normal gradient magnitude
- 0.1x normal gradient magnitude (vanishing)
- 10x normal input scale
- 10x normal batch size variance
- NaN/Inf at any position in any tensor
- Mixed precision edge cases
- Zero-size tensors (empty batches)
- Duplicate/NaN labels
- 100K+ sequence lengths (attention)
- 10K+ layer depth (extreme residual)

---

## Paper Scraping — Research Pipeline

### Priority papers (July 5):

1. **Mamba/SSM stability** — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
   → Test SSM-specific failure modes: selective scan divergence

2. **KAN training stability** — "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
   → Spline basis function numerical stability

3. **MoE training pitfalls** — "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (Zoph et al., 2022)
   → Load balancing loss interaction, expert collapse detection

4. **Mixed precision failure analysis** — "Mixed Precision Training" (Micikevicius et al., 2018)
   → Loss scaling, gradient underflow patterns

5. **GNN gradient analysis** — "Understanding GNN Training Dynamics" (multiple authors)
   → Oversmoothing as vanishing, oversquashing as bottleneck

6. **Diffusion model instabilities** — "Analyzing Diffusion Model Training Stability" (2023-2024)
   → Timestep-specific gradient variance

7. **torch.compile edge cases** — PyTorch issue tracker + Dynamo papers
   → Graph breaks, guard failures, recompilation triggers

8. **FlashAttention numerical notes** — Dao et al. technical reports
   → Block-sparse gradient approximation errors
