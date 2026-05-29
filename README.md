# NeuralDBG

A causal inference engine for deep learning training that provides **structured explanations** of neural network training failures. Understand *why* your model failed during training through semantic analysis and abductive reasoning, not raw tensor inspection.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/neuraldbg.svg)](https://pypi.org/project/neuraldbg/)
[![CI](https://github.com/LambdaSection/NeuralDBG/actions/workflows/ci.yml/badge.svg)](https://github.com/LambdaSection/NeuralDBG/actions/workflows/ci.yml)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://lambdasection.github.io/NeuralDBG/)

## Overview

NeuralDBG treats training as a **semantic trace of learning dynamics** rather than a black box. It extracts meaningful events and provides causal hypotheses about training failures, enabling researchers to:

- **Identify gradient health transitions** (stable -> vanishing/saturated)
- **Detect activation regime shifts** (normal -> saturated/dead)
- **Detect optimizer instability** (loss plateaus, spikes, divergence)
- **Catch data anomalies** (NaN, Inf, distribution shifts)
- **Track propagation of instabilities** through network layers
- **Generate ranked causal explanations** for training failures

Unlike traditional monitoring tools (TensorBoard, Weights & Biases), NeuralDBG focuses on **causal inference** rather than metric tracking.

## Neural Suite

| Component | Availability | Role |
|-----------|--------------|------|
| **NeuralDBG** | Public ([PyPI](https://pypi.org/project/neuraldbg/)) | Causal diagnostics in your training loop |
| **Diagnostic Workspace** | Private beta | Visual causal graphs and hypothesis explorer |
| **Neural Agent** | Private beta | Auto-remediation from causal hypotheses |

Request early access: [open an issue](https://github.com/LambdaSection/NeuralDBG/issues/new?labels=suite-access) with label `suite-access`.

## Try it in 60 seconds (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LambdaSection/NeuralDBG/blob/main/notebooks/quickstart.ipynb)

Or locally: `python examples/quickstart_interactive.py`

## Public benchmark

Reproducible causal accuracy on synthetic failures ([details](benchmark_public/README.md)):

```bash
python -m benchmark_public.run
```

Latest results: [benchmark_public/results.json](benchmark_public/results.json)

## Why NeuralDBG?

| Feature | TensorBoard / W&B | NeuralDBG |
|---|---|---|
| **What it shows** | Graphs of loss/accuracy over time | **Why** the loss spiked or vanished |
| **Diagnosis** | Manual inspection of curves | **Automated causal hypotheses** |
| **Actionable?** | You guess the fix | Suggests root causes (LR, Init, Data) |
| **Integration** | Separate dashboard | **One line of code** in your loop |
| **Privacy** | Data sent to cloud | **100% Local** (unless you opt-in) |

> "TensorBoard tells you *when* it failed. NeuralDBG tells you *why*."

## Key Features

- **Semantic Event Extraction**: Detects meaningful transitions in training dynamics
- **Causal Compression**: Identifies first occurrences and propagation patterns
- **Post-Mortem Reasoning**: Provides ranked hypotheses about failure causes
- **Optimizer Instability Detection**: Tracks loss plateaus, spikes, and divergence
- **Data Anomaly Detection**: Catches NaN, Inf, and distribution shifts in inputs
- **Event Collapsing**: Merges sequential events into summary traces
- **Compiler-Aware**: Operates at module boundaries to survive torch.compile
- **Non-Invasive**: Wraps existing PyTorch training loops without code changes
- **Minimal API**: Focused on explanations, not raw data dumps
- **Diagnostic package export**: JSON export for the Neural Suite visualizer (private beta)

## Quick Start

### Installation

```bash
pip install neuraldbg
```

### Basic Usage

```python
import torch
import torch.nn as nn
from neuraldbg import NeuralDbg

# Your existing model and training setup
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Wrap your training loop
with NeuralDbg(model) as dbg:
    for step, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# After training failure, query for explanations
explanations = dbg.explain_failure()
print(explanations[0])  # "Gradient vanishing originated in layer 'linear1' at step 234..."
```

### Inference API

```python
# Get ranked causal hypotheses for the failure
hypotheses = dbg.get_causal_hypotheses()

# Query specific causal chains
chain = dbg.trace_causal_chain('vanishing_gradients')

# Check for coupled failures
couplings = dbg.detect_coupled_failures()

# Export diagnostic package (JSON) for Neural Suite visualizer
dbg.export_aquarium_package('debug_session.json')
```

### Optimizer Instability Detection

```python
with NeuralDbg(model) as dbg:
    for step in range(num_steps):
        dbg.step = step
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        dbg.record_loss(loss.item())
        optimizer.step()

# Detect loss plateaus, spikes, or divergence
hypotheses = dbg.explain_failure("optimizer_instability")
for h in hypotheses:
    print(h.description)
```

### Data Anomaly Detection

Data anomalies (NaN, Inf, distribution shifts) are detected automatically from layer inputs during the forward pass:

```python
with NeuralDbg(model) as dbg:
    # ... training loop ...
    pass

hypotheses = dbg.explain_failure("data_anomaly")
for h in hypotheses:
    print(h.description)  # "NaN values detected in input to layer 'linear1'..."
```

## Supported Architectures

NeuralDBG has been validated across 9 architectures:

| Architecture | Failure Modes Tested |
|---|---|
| Transformer (nanoGPT) | Attention collapse, NaN softmax, LR warmup |
| GANs (DCGAN) | Vanishing, exploding, NaN injection |
| LLM fine-tuning (LoRA) | Catastrophic forgetting, loss spikes |
| Diffusion (DDPM) | NaN UNet, exploding gradients |
| LSTM / Time Series | Vanishing recurrent gradients |
| GNN (GCN/GAT) | Oversmoothing, deep GNN |
| RL (PPO-style) | Policy collapse, value explosion |
| torch.compile | Dynamo graph compatibility |
| DataParallel | Multi-GPU hook integrity |

## Supported Failure Types

| Failure Type | Description |
|---|---|
| `vanishing_gradients` | Root cause + saturation coupling |
| `exploding_gradients` | First layer to explode |
| `dead_neurons` | Neuron death in activation layers |
| `saturated_activations` | Activation saturation patterns |
| `optimizer_instability` | Loss plateaus, spikes, divergence |
| `data_anomaly` | NaN/Inf/distribution shift in inputs |

## Architecture

### Core Components

- **Semantic Event Extractor**: Detects meaningful transitions in learning dynamics
- **Causal Compressor**: Identifies patterns and propagation in training failures
- **Post-Mortem Reasoner**: Generates ranked hypotheses about failure causes
- **Compiler-Aware Monitor**: Operates at safe boundaries for optimization compatibility

### Event Types

| Event Type | Source | Detects |
|---|---|---|
| `gradient_health_transition` | Backward hooks | Vanishing, exploding, saturated gradients |
| `activation_regime_shift` | Forward hooks | Dead neurons, saturated activations |
| `optimizer_instability` | `record_loss()` | Loss plateaus, spikes, divergence |
| `data_anomaly` | Forward hooks (inputs) | NaN, Inf, distribution shifts |

## Editions

| Edition | Package | License | Features |
|---|---|---|---|
| **Core** | `pip install neuraldbg` | MIT | Hooks, events, export JSON, basic heuristics |
| **Engine** | `pip install neuraldbg-engine` | Proprietary | Full causal inference, detailed hypotheses, coupling detection |

The Core edition works standalone with basic heuristic fallbacks. Install the Engine for advanced causal reasoning.

## Target Users

- **ML Researchers** seeking causal explanations for training failures
- **PhD Students** analyzing learning dynamics in novel architectures
- **Research Engineers** understanding optimization instabilities

## Limitations

- PyTorch only
- Focus on semantic events, not tensor inspection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Developer Setup

```bash
make bootstrap
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Documentation

- [Landing page](https://lambdasection.github.io/NeuralDBG/) — branding and suite overview
- [PyTorch support matrix](docs/PYTORCH_SUPPORT.md)
- [CHANGELOG.md](CHANGELOG.md) - Version history and notable changes
- [logic_graph.md](logic_graph.md) - System architecture and data flow
- [docs/PHASE2_DOGFOODING.md](docs/PHASE2_DOGFOODING.md) - Detailed dogfooding scenarios

## Citation

If you use NeuralDBG in your research, please cite:

```bibtex
@misc{neuraldbg2026,
  title={NeuralDBG: A Causal Inference Engine for Deep Learning Training Dynamics},
  author={SENOUVO Jacques-Charles Gad},
  year={2026},
  url={https://github.com/LambdaSection/NeuralDBG}
}
```
