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

- **Identify gradient health transitions** (stable -> vanishing/saturated)
- **Detect activation regime shifts** (normal -> saturated/dead)
- **Detect optimizer instability** (loss plateaus, spikes, divergence)
- **Catch data anomalies** (NaN, Inf, distribution shifts)
- **Track propagation of instabilities** through network layers
- **Generate ranked causal explanations** for training failures

Unlike traditional monitoring tools (TensorBoard, Weights & Biases), NeuralDBG focuses on **causal inference** rather than metric tracking.

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

Equivalent shortcuts via `Makefile`:

```bash
make build
make up
make shell
```

Run tests inside Docker:

```bash
docker-compose run --rm neuraldbg-dev bash -lc "pytest"
```

Or:

```bash
make test-docker
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

### Optimizer Instability Detection

```python
with NeuralDbg(model) as dbg:
    for step in range(num_steps):
        dbg.step = step
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()

        # Feed loss values for optimizer instability detection
        dbg.record_loss(loss.item())

        optimizer.step()

# Detect loss plateaus, spikes, or divergence
hypotheses = dbg.explain_failure("optimizer_instability")
for h in hypotheses:
    print(h.description)  # "Loss spike detected at step 50..."
```

### Data Anomaly Detection

Data anomalies (NaN, Inf, distribution shifts) are detected automatically
from layer inputs during the forward pass -- no extra API call needed:

```python
with NeuralDbg(model) as dbg:
    # ... training loop ...
    pass

# Check for data issues
hypotheses = dbg.explain_failure("data_anomaly")
for h in hypotheses:
    print(h.description)  # "NaN values detected in input to layer 'linear1'..."
```

### Event Collapsing

Compress sequential events in the same layer into summary traces:

```python
# Get compressed event timeline
collapsed = dbg._collapse_events()
print(f"{len(dbg.events)} raw events -> {len(collapsed)} collapsed")
```

## Architecture

### Core Components

- **Semantic Event Extractor**: Detects meaningful transitions in learning dynamics
- **Causal Compressor**: Identifies patterns and propagation in training failures
- **Post-Mortem Reasoner**: Generates ranked hypotheses about failure causes
- **Compiler-Aware Monitor**: Operates at safe boundaries for optimization compatibility

### Event Types

| Event Type | Source | Detects |
|------------|--------|---------|
| `gradient_health_transition` | Backward hooks | Vanishing, exploding, saturated gradients |
| `activation_regime_shift` | Forward hooks | Dead neurons, saturated activations |
| `optimizer_instability` | `record_loss()` | Loss plateaus, spikes, divergence |
| `data_anomaly` | Forward hooks (inputs) | NaN, Inf, distribution shifts |

### Event Structure

Each semantic event represents:
- Transition type (gradient_health, activation_regime, optimizer_instability, data_anomaly)
- Layer/parameter identifier
- Step range of occurrence
- Confidence score
- Causal metadata (propagation patterns, coupled failures)

## Target Users

- **ML Researchers** seeking causal explanations for training failures
- **PhD Students** analyzing learning dynamics in novel architectures
- **Research Engineers** understanding optimization instabilities

*Not intended for production monitoring, metric tracking, or no-code users.*

## Supported Failure Types

- `vanishing_gradients` -- Root cause + saturation coupling
- `exploding_gradients` -- First layer to explode
- `dead_neurons` -- Neuron death in activation layers
- `saturated_activations` -- Activation saturation patterns
- `optimizer_instability` -- Loss plateaus, spikes, divergence (with gradient cross-reference)
- `data_anomaly` -- NaN/Inf/distribution shift in inputs

## Limitations (MVP Scope)

- PyTorch only
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
- [GOOGLE_DOCS_SYNC.md](GOOGLE_DOCS_SYNC.md) - Daily SESSION_SUMMARY sync to Google Docs

## Google Docs Daily Sync

You can automate daily publication of `SESSION_SUMMARY.md` to a Google Doc:

1. Install optional automation deps:
```bash
pip install -e .[automation]
```
2. Configure:
   - `GOOGLE_DOC_ID`
   - `GOOGLE_SERVICE_ACCOUNT_FILE` or `GOOGLE_SERVICE_ACCOUNT_JSON`
3. Run:
```bash
python scripts/publish_session_summary_to_gdocs.py --source SESSION_SUMMARY.md --mode append
```

For GitHub Actions-based daily sync, see `.github/workflows/publish-summary-to-google-docs.yml`.

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
