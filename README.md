# NeuralDBG

A causal inference engine for deep learning training that provides **structured explanations** of neural network training failures. Understand *why* your model failed during training through semantic analysis and abductive reasoning, not raw tensor inspection.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/Lemniscate-world/Neural/workflows/Build/badge.svg)](https://github.com/Lemniscate-world/Neural/actions)
[![CodeQL](https://github.com/Lemniscate-world/Neural/workflows/CodeQL/badge.svg)](https://github.com/Lemniscate-world/Neural/actions)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Overview

NeuralDBG treats training as a **semantic trace of learning dynamics** rather than a black box. It extracts meaningful events and provides causal hypotheses about training failures, enabling researchers to:

- **Identify gradient health transitions** (stable → vanishing/saturated)
- **Detect activation regime shifts** (normal → saturated/dead)
- **Track propagation of instabilities** through network layers
- **Generate ranked causal explanations** for training failures

Unlike traditional monitoring tools (TensorBoard, Weights & Biases), NeuralDBG focuses on **causal inference** rather than metric tracking.

## Key Features

- **Semantic Event Extraction**: Detects meaningful transitions in training dynamics
- **Causal Compression**: Identifies first occurrences and propagation patterns
- **Post-Mortem Reasoning**: Provides ranked hypotheses about failure causes
- **Compiler-Aware**: Operates at module boundaries to survive torch.compile
- **Non-Invasive**: Wraps existing PyTorch training loops without code changes
- **Minimal API**: Focused on explanations, not raw data dumps

## Quick Start

### Installation

```bash
pip install neuraldbg
```

### Docker Development (Hermetic Workspace)

Use Docker to keep a reproducible local environment across machines and contributors.

```bash
# Build image
docker-compose build

# Start the dev container (one-command startup)
docker-compose up -d

# Open a shell in the running workspace
docker-compose exec neuraldbg-dev bash
```

Run tests inside Docker:

```bash
docker-compose run --rm neuraldbg-dev bash -lc "pytest"
```

Persistent volumes are mounted to:
- `/data` (host: `./data`)
- `/models` (host: `./models`)
- `/outputs` (host: `./outputs`)

Stop containers:

```bash
docker-compose down
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
        optimizer.step()

        # Events are extracted automatically

# After training failure, query for explanations
explanations = dbg.explain_failure()
print(explanations[0])  # "Gradient vanishing originated in layer 'linear1' at step 234, likely due to LR × activation mismatch (confidence: 0.87)"
```

### Inference API

```python
# Get ranked causal hypotheses for the failure
hypotheses = dbg.get_causal_hypotheses()

# Query specific causal chains
chain = dbg.trace_causal_chain('vanishing_gradients')

# Check for coupled failures
couplings = dbg.detect_coupled_failures()
```

## Architecture

### Core Components

- **Semantic Event Extractor**: Detects meaningful transitions in learning dynamics
- **Causal Compressor**: Identifies patterns and propagation in training failures
- **Post-Mortem Reasoner**: Generates ranked hypotheses about failure causes
- **Compiler-Aware Monitor**: Operates at safe boundaries for optimization compatibility

### Event Structure

Each semantic event represents:
- Transition type (gradient_health, activation_regime, optimizer_stability)
- Layer/parameter identifier
- Step range of occurrence
- Confidence score
- Causal metadata (propagation patterns, coupled failures)

## Target Users

- **ML Researchers** seeking causal explanations for training failures
- **PhD Students** analyzing learning dynamics in novel architectures
- **Research Engineers** understanding optimization instabilities

*Not intended for production monitoring, metric tracking, or no-code users.*

## Limitations (MVP Scope)

- PyTorch only
- Single causal question: "Why did gradients vanish here?"
- Focus on semantic events, not tensor inspection
- Command-line interface only
- Compiler-aware (torch.compile compatible)

## Contributing

This is an MVP focused on proving the concept of causal inference for training dynamics. Contributions should align with the core mission of providing structured explanations for training failures.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## Documentation

- [PROJECTS.md](PROJECTS.md) - Roadmap Projets A & B (Projet A dans Quant-Search, B ici)
- [CHANGELOG.md](CHANGELOG.md) - Version history and notable changes
- [logic_graph.md](logic_graph.md) - System architecture and data flow

## Citation

If you use NeuralDBG in your research, please cite:

```bibtex
@misc{neuraldbg2025,
  title={NeuralDBG: A Causal Inference Engine for Deep Learning Training Dynamics},
  author={SENOUVO Jacques-Charles Gad},
  year={2025},
  url={https://github.com/Lemniscate-world/Neural}
}
```
