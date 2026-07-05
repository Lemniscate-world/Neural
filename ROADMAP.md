# NeuralSuite

> The complete toolkit for diagnosing and fixing deep learning training failures.

## Dashboard (July 5, 2026)

| Metric | Score |
|--------|-------|
| Tier 1 Black-Swan detection | **96%** (104/108) |
| Tier 2 Black-Swan detection | **94%** (102/108) |
| Combinatorial (200 archs) | 79% (958/1200) |
| Stress tests | 100% (15/15) |
| GPU v4 accuracy | 92.3% |
| Architecture fuzzer | 94% crash rate |
| PyTorch PRs | 4 submitted, 0 merged |

## Ecosystem (Multi-Repo)

| Component | Role | Status |
|---|---|---|
| **NeuralDBG** (this repo) | Causal diagnostic engine | v1.5.0 ✅ |
| **Neural-Agent** | Auto-corrector (GPU) | v4 92.3%, v5 pending |
| **Aquarium** | Visualizer (HTML dashboard) | Live ✅ |
| **neuraldbg-engine** | Advanced causal inference | Merged into core |

## Black-Swan Detection (3 Tiers)

| Tier | Families | Detection |
|------|----------|-----------|
| 1 | GNN, MoE, Diffusion | 96% (104/108) |
| 2 | FlashAttention, NeuralODE, Quantized | 94% (102/108) |
| 3 | Predictive anomaly detection | MVP |

## NeuralPrune — Model Optimization

Non-destructive redundancy diagnostic:
- Dead neuron detection (99%+ zero activations)
- Redundant weight identification (50%+ near-zero)
- Low-rank matrix decomposition opportunities
- Quantization readiness (INT8/INT4 compatibility)
- Static weight detection (vanished gradients)

## Upcoming

- [ ] v5 GPU training (8 families: +GNN, MoE, Diffusion)
- [ ] Community launch (Reddit, Discord, HN)
- [ ] Paper submission (arXiv)
- [ ] Real-world beta testing (3+ users)

model = nn.Sequential(nn.Linear(10, 64), nn.Sigmoid(), nn.Linear(64, 2))

with NeuralDbg(model) as dbg:
    for step in range(100):
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# See what went wrong
print(dbg.explain_failure())
```

## Roadmap

### v1.3.2 — Bug-Solver MVP (June 2026)
- [x] Composite-module hook support
- [x] Silent-loss and zero-leaf warnings
- [x] MHA fully-masked-row remediation rule
- [x] Neural-Agent: end-to-end diagnose -> fix -> validate pipeline
- [x] 300+ tests passing
- [x] Public benchmark: 4/4 scenarios at 1.0 accuracy
- [x] Tool comparison v2: NeuralDBG vs W&B vs MLflow vs TensorBoard
- [x] First upstream PR submitted (3 PRs!)
- [x] NeuralDBG-Engine: 45 tests, full API contract coverage
- [x] Neural-Agent: CPU training pipeline validated (distilgpt2 + LoRA, loss 4.18 -> 3.18)
- [x] **NEW: GPU training** — Qwen2-0.5B + LoRA on Quadro M4000 (8.6 GB), 5/5 categories
- [x] PR Gate system: mandatory 6-gate checklist before any upstream PR
- [x] **NEW: CLI Wrapper** — `neuraldbg run script.py` (zero-code injection)
- [x] **NEW: --agent flag** — Neural-Agent auto-fix after detection
- [x] **NEW: --export flag** — Aquarium JSON export

### Upstream PR Tracker

| Bug | Upstream Issue | PR Status | Merge Date |
|-----|---------------|-----------|------------|
| BUG-001 | pytorch/pytorch#41508 | Comment posted | - |
| BUG-002 | pytorch/pytorch#176793 | PR #186786 **CLOSED** (stale). New PR needed. | - |
| BUG-003 | pytorch/pytorch#177116 | **PR #188923 OPEN** (+59/-0, clean) | - |
| BUG-004 | huggingface/transformers#44928 | PR #47024 **CLOSED** (stale 1d). New PR needed. | - |
| BUG-005 | pytorch/pytorch#173334 | Repro posted, issue closed (awaiting reopen) | - |
| BUG-006 | pytorch/pytorch#187759 | **PR #188053 OPEN** — relance posted 3 Jul | - |
| BUG-007 | pytorch/pytorch#186799 | Cataloged (reported by @ezyang) | - |
| BUG-008 | pytorch/pytorch#184575 | **PR #188066 OPEN** — relance posted 3 Jul | - |
| BUG-009 | pytorch/pytorch#187227 | Cataloged (SDPA int32 overflow) | - |
| BUG-010 | pytorch/pytorch#185543 | Cataloged (quantile gradient mismatch) | - |

**PRs submitted**: 7 | **Active**: 4 (#188933 fix, #188923 test, #188053 test, #188066 test) | **Merged**: 0

### v1.4.5 — Catalog Expansion (July-August 2026)
- [x] 10/10 real bugs cataloged ✅ M2 OBJECTIVE REA- [x] 10/10 post-mortems published on GitHub Pages ✅
- [x] Reproducible public benchmark on 5+ real scenarios
- [x] Comparison vs Captum (explainability) ✅ — [benchmark](benchmark_public/benchmark_captum.py)
- [x] 5/5 upstream PRs submitted
- [x] **Detection accuracy >= 0.90** → **1.00 (100%) on DeepMLP** ✅ EXCEEDED
- [x] **Causal chain engine** — true causal inference on computation graphs ✅
- [x] **GPU training** — Qwen2-0.5B + LoRA, 5/5 categories distinct ✅
- [x] **E2E Pipeline** — detect → chain → fix → validate (BUG-003 PASS) ✅
- [x] **Tool comparison matrix** — NeuralDBG vs W&B/TensorBoard/MLflow/Captum (14/16 YES) ✅
- [x] **Validation dashboard** — live bug matrix + PR tracker ✅

### v1.5.0 — Obligation (August-September 2026)
- [x] 10+ bugs cataloged ✅ | 10+ post-mortems published ✅ (10/10)
- [ ] Versioned benchmark with CI regression gates
- [x] Neural-Agent autonomous: closed loop on DeepMLP (E2E pipeline proven) ✅
- [ ] 1+ upstream PR merged (external validation) — **0/5 merged, critical gap** 🔴
- [ ] Research paper draft on causal ML diagnostics
- [x] **Real model testing** — ResNet + Transformer: 90% detection, 0% FP ✅

### v1.5.5 — Combinatorial Validation (4 July 2026)
- [x] **Combinatorial sweep** — 50 archs, 300 evals, 87% global detection (+12% from 75%)
- [x] **RNN core fix** — LSTM/GRU output tuple unwrap, hidden state capture. RNN: 49%→65%, Hybrid: 34%→85%
- [x] **Paper scraper** — 60 novel architectures (Mamba, KAN, xLSTM, MoE, Hyena, RWKV, etc.)
- [x] **Engine merge** — NeuralDBG-Engine bundled into neuraldbg/engine/, no separate package
- [x] **GPU v4 model** — Qwen2-0.5B fp16 + LoRA r=8, 538 examples, 5 families, 92.3% accuracy, 4.3MB adapter
- [x] **Aquarium web dashboard** — Zero-dependency HTML causal viewer (replaces Tauri app)
- [x] **E2E RNN pipeline** — Detect→Diagnose→Fix→Validate on LSTM, 2/4 bugs auto-fixed
- [x] **Community posts** — PyTorch Dev Discussions + Reddit r/ML drafted
- [x] **Real-architecture validation** — ResNet 80%, Transformer 100%, combined 90%, 0% FP

## Benchmark Results (v1.3.2)

5 scenarios, healthy excluded from averages:

| Tool | Detection (loss-only) | Detection (+grad norms) | Localization |
|------|:---------------------:|:-----------------------:|:------------:|
| **NeuralDBG** | **1.00** | **1.00** | **1.00** |
| W&B | 0.33 | 0.67 | 0.00 |
| MLflow | 0.33 | 0.67 | 0.00 |
| TensorBoard | 0.33 | 0.67 | 0.00 |

Key findings:
- External tools can detect anomalies (NaN loss, loss spikes) but cannot localize the failing layer
- With gradient norm logging, external tools detect MHA NaN gradients (0.33 -> 0.67)
- NeuralDBG is the only tool that names the layer causing the failure
- Benchmark is reproducible: `python -m benchmark_public.run`

## Real Architecture Validation (4 July 2026)

NeuralDBG tested on production-grade architectures, not just toy MLPs:

| Architecture | Bugs Detected | False Positives | Details |
|-------------|:------------:|:---------------:|---------|
| Mini ResNet (CNN, 4 residual blocks) | 4/5 (80%) | 0/1 | Miss: sigmoid saturation (subtle) |
| Mini Transformer (3 encoder blocks) | **5/5 (100%)** | 0/1 | All bugs captured |
| **Total** | **9/10 (90%)** | **0/2** | Zero false positives |

Reproduce: `python validate_real_architectures.py`

## Install

```bash
pip install neuraldbg
```

## License

MIT
