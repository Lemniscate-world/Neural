# Neural DSL Documentation

## What's New in v0.2.6

### Key Improvements
- **Enhanced Dashboard UI**: Improved NeuralDbg dashboard with a more aesthetic dark theme design (#452)
- **Advanced HPO Examples**: Added comprehensive examples for complex hyperparameter optimization configurations (#448)
- **Blog Support**: Infrastructure for blog content with markdown support and Dev.to integration (#445)
- **Improved Error Reporting**: Planning enhanced error context in validation messages with more precise line/column information (#459)
- **CLI Version Display**: Updated version command to dynamically fetch package version (#437)

### Example: Enhanced Dashboard UI
```bash
# Start the NeuralDbg dashboard with the new dark theme
neuraldbg --theme dark

# Or use the debug command with visualization
neural debug my_model.neural --gradients --theme dark
```

### Example: Advanced Nested HPO Configuration
```yaml
network AdvancedHPOExample {
  input: (28, 28, 1)
  layers:
    # Convolutional layers with HPO parameters
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    # Another conv block with HPO
    Conv2D(filters=HPO(choice(64, 128)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    # Flatten and dense layers
    Flatten()
    Dense(HPO(choice(128, 256, 512)), activation="relu")
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")

  # Advanced optimizer configuration with HPO
  optimizer: SGD(
    learning_rate=ExponentialDecay(
      HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
      1000,                              # Decay steps
      HPO(range(0.9, 0.99, step=0.01))   # Decay rate
    ),
    momentum=HPO(range(0.8, 0.99, step=0.01))
  )

  # Training configuration with HPO
  train {
    epochs: 20
    batch_size: HPO(choice(32, 64, 128))
    validation_split: 0.2
    search_method: "bayesian"  # Use Bayesian optimization
  }
}
```

## What's New in v0.2.5

### Key Improvements
- **Multi-Framework HPO Support**: Seamless hyperparameter optimization across PyTorch and TensorFlow
- **Enhanced Optimizer Handling**: Improved parsing and validation of optimizer configurations
- **Precision & Recall Metrics**: Comprehensive metrics reporting in training loops
- **Error Message Improvements**: More detailed error messages with line/column information
- **No-Quote Syntax**: Cleaner syntax for optimizer parameters without quotes

### Example: Advanced HPO with Learning Rate Schedules
```yaml
optimizer: SGD(
  learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
    1000,                              # Decay steps
    HPO(range(0.9, 0.99, step=0.01))   # Decay rate
  ),
  momentum=HPO(range(0.8, 0.99, step=0.01))
)
```

## Table of Contents
- [Syntax Reference](#syntax-reference)
- [Core Components](#core-components)
- [Validation Rules](#validation-rules)
- [CLI Reference](#cli-reference)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Enhanced Dashboard UI (v0.2.6+)](#enhanced-dashboard-ui-v026)
- [Blog Support (v0.2.6+)](#blog-support-v026)
- [Development Guide](#development-guide)

---

## Syntax Reference

### Network Structure
```yaml
network <ModelName> {
  input: <Shape>                # e.g., (224, 224, 3) or multiple inputs
  layers:
    <LayerType>(<Parameters>)   # Supports ordered and named params
    <LayerType>*<Count>         # Layer repetition syntax
  loss: <LossFunction>
  optimizer: <Optimizer>
  train {
    epochs: <Int>
    batch_size: <Int|HPO-range>
    validation_split: <0.0-1.0>
    search_method: "random"     # Hyperparameter optimization
  }
  execution {
    device: <String>            # "cpu", "cuda", "tpu"
  }
}
```

### Parameter Types
```yaml
# Ordered parameters
Conv2D(32, (3,3), "relu")

# Named parameters
TransformerEncoder(num_heads=8, ff_dim=512)

# HPO parameters
Dense(HPO(choice(128, 256, 512)))

# Device placement
LSTM(units=128) @ "cuda:0"
```

---

## Core Components

### Layer Types
| Category         | Layers                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Convolution**  | `Conv1D`, `Conv2D`, `DepthwiseConv2D`, `SeparableConv2D`              |
| **Recurrent**    | `LSTM`, `GRU`, `Bidirectional`, `ConvLSTM2D`                          |
| **Transformer**  | `TransformerEncoder`, `TransformerDecoder`                            |
| **Regularization**| `Dropout`, `SpatialDropout2D`, `GaussianNoise`, `BatchNormalization` |
| **Utility**      | `Flatten`, `Lambda`, `TimeDistributed`, `ResidualConnection`          |

---

## Validation Rules

### Parameter Constraints
| Parameter           | Rule                                  | Error Example              |
|---------------------|---------------------------------------|----------------------------|
| `num_heads`         | Must be > 0                          | `num_heads=0` → ERROR      |
| `filters`           | Positive integer                     | `filters=-32` → ERROR      |
| `kernel_size`       | Tuple of positive integers           | `(0,3)` → ERROR            |
| `rate`              | 0 ≤ value ≤ 1                        | `Dropout(1.2)` → ERROR     |
| `validation_split`  | 0 ≤ value ≤ 1                        | `1.1` → ERROR              |
| `device`            | Valid device identifier              | `device:"npu"` → CRITICAL  |

### Type Coercions
- Float → Int conversion with warning (e.g., `Dense(256.0)`)
- Automatic tuple wrapping for single values

---

## CLI Reference

### Command Overview
```bash
# Core Commands
neural compile <file> [--backend tensorflow|pytorch] [--hpo]
neural run <file> [--device cpu|cuda]
neural debug <file> [--gradients] [--dead-neurons] [--step]

# Analysis & Visualization
neural visualize <file> [--format png|svg|html]
neural profile <file> [--memory] [--latency]

# Project Management
neural clean  # Remove generated files
neural version  # Show version and dependencies info
```

### Key Options
- `--dry-run`: Validate without code generation
- `--hpo`: Enable hyperparameter optimization
- `--step`: Interactive debugging mode
- `--port`: Specify GUI port for no-code interface
- `--theme`: Set dashboard theme (light/dark, v0.2.6+)

---

## Error Handling

### Severity Levels
| Level     | Description                          | Example Case                      |
|-----------|--------------------------------------|-----------------------------------|
| CRITICAL  | Fatal configuration error           | Invalid device specification      |
| ERROR     | Structural/model error              | Missing required parameter        |
| WARNING   | Non-fatal issue                     | Type coercion                     |
| INFO      | Diagnostic message                  | Shape propagation update          |

### Example Messages
```text
CRITICAL at line 5, column 12:
Invalid device 'npu' - must be one of [cpu, cuda, tpu]

ERROR at line 12, column 8:
Conv2D kernel_size requires positive integers. Got (0,3)

WARNING at line 8, column 15:
Implicit conversion of 256.0 to integer 256
```

### Planned Error Message Improvements (v0.2.6)

Version 0.2.6 plans to improve error messages with more context and better formatting (#459):

```text
# Current error format
ERROR at line 15, column 10:
Dense units must be positive integers. Got -128

# Planned improvements for future releases
# ERROR at line 15, column 10:
# Dense units must be positive integers. Got -128
# Context: In network 'MNISTClassifier', layer 2
# Suggestion: Use a positive integer value
```

Planned error message improvements include:
- Context information about where the error occurred
- Suggestions for fixing common issues
- Better handling of nested structures

### Enhanced Error Messages (v0.2.5)

Version 0.2.5 includes improved error messages with more context and better formatting:

```text
ERROR at line 15, column 10:
Dense units must be positive integers. Got -128

ERROR at line 22, column 14:
Dropout rate must be between 0 and 1. Got 1.5

ERROR at line 30, column 18:
TransformerEncoder num_heads must be positive. Got 0

WARNING at line 42, column 22:
HPO parameter 'learning_rate' should use log_range for better optimization
```

Error messages now include:
- Precise line and column numbers
- Clear descriptions of the validation rule that was violated
- The actual value that caused the error
- Suggestions for fixing common issues

---

## Examples

### Vision Transformer
```yaml
network ViT {
  input: (224, 224, 3)
  layers:
    Conv2D(64, (7,7), strides=2) @ "cuda:0"
    TransformerEncoder() * 12     # 12 transformer blocks
    GlobalAveragePooling2D()
    Dense(1000, "softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=1e-4)
  train {
    epochs: 300
    batch_size: HPO(range(128, 512, step=64))
  }
}
```

### Hyperparameter Optimization
```yaml
network HPOExample {
  layers:
    Dense(HPO(choice(128, 256, 512)))
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
  optimizer: Adam(
    learning_rate=HPO(log_range(1e-4, 1e-2))
  )
  train {
    epochs: 100
    search_method: "bayesian"
  }
}
```

### Parameter Types
```yaml
# Categorical Choice
HPO(choice(128, 256, 512))      # Select from discrete values

# Range with Step
HPO(range(0.3, 0.7, step=0.1))  # Linear range with step size

# Logarithmic Range
HPO(log_range(1e-4, 1e-2))      # Log-scale range for learning rates
```

### Supported Parameters
| Parameter | HPO Type | Example |
|-----------|----------|---------|
| Dense units | `choice` | `Dense(HPO(choice(64, 128, 256)))` |
| Dropout rate | `range` | `Dropout(HPO(range(0.3, 0.7, step=0.1)))` |
| Learning rate | `log_range` | `Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))` |

### Validation Rules
- Dense units must be positive integers
- Dropout rate must be between 0 and 1
- Learning rate must be positive
- Step size must be provided for range type

### HPO Parameters Updates (v0.2.6)
- **Advanced Examples**: Added comprehensive examples demonstrating complex HPO configurations (#448):
  - Multiple HPO parameters within the same layer
  - HPO parameters in both optimizer and learning rate schedules
  - Batch size optimization alongside model parameters
- **Planned Error Handling Improvements**: Planning better validation and error reporting for HPO parameters (#459):
  - More precise line/column information in error messages
  - Detailed context about parameter constraints
  - Validation for complex nested configurations

### HPO Parameters Updates (v0.2.5)
- **Multi-Framework Support**: HPO now works seamlessly across both PyTorch and TensorFlow backends.
- **Optimizer Parameters**: All optimizer parameters now support HPO, including:
  - `learning_rate` with `HPO(log_range(1e-4, 1e-2))` (#434)
  - `beta_1` and `beta_2` for Adam
  - `momentum` for SGD
- **Learning Rate Schedules**: HPO parameters can be nested within learning rate schedules.
- **String Representation**: Improved handling of scientific notation (e.g., `1e-4` vs `0.0001`).
- **No-Quote Syntax**: Parameters can be specified without quotes for cleaner syntax.

### Examples
#### Basic HPO Example
```yaml
network HPOExample {
  input: (28, 28, 1)
  layers:
    Dense(HPO(choice(128, 256)))
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")
  optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
  train {
    epochs: 10
    search_method: "random"
  }
}
```

#### Advanced HPO with Learning Rate Schedules
```yaml
network AdvancedHPO {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3))
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(HPO(choice(128, 256, 512)))
    Output(10, "softmax")
  optimizer: SGD(
    learning_rate=ExponentialDecay(
      HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
      1000,                              # Decay steps
      HPO(range(0.9, 0.99, step=0.01))   # Decay rate
    ),
    momentum=HPO(range(0.8, 0.99, step=0.01))
  )
  train {
    epochs: 20
    batch_size: HPO(choice(32, 64, 128))
    search_method: "bayesian"
  }
}

---

## Training Configuration

### Basic Setup
```yaml
network MyModel {
    train {
        epochs: 100
        batch_size: 32
        validation_split: 0.2
        search_method: "bayesian"  # For HPO
    }
}
```

### Optimizer Configuration
```yaml
# Basic optimizer configuration
optimizer: Adam(
    learning_rate=HPO(log_range(1e-4, 1e-2)),
    beta_1=0.9,
    beta_2=0.999
)

# Learning rate schedules
optimizer: SGD(
    learning_rate=ExponentialDecay(0.1, 1000, 0.96),
    momentum=0.9
)

# Learning rate schedules with HPO
optimizer: SGD(
    learning_rate=ExponentialDecay(HPO(range(0.05, 0.2, step=0.05)), 1000, HPO(range(0.9, 0.99, step=0.01)))
)
```

### Learning Rate Schedules
Learning rate schedules allow you to dynamically adjust the learning rate during training. They can be specified directly in the `learning_rate` parameter of optimizers.

```yaml
# ExponentialDecay schedule
learning_rate=ExponentialDecay(0.1, 1000, 0.96)

# With HPO parameters
learning_rate=ExponentialDecay(HPO(range(0.05, 0.2, step=0.05)), 1000, 0.96)
```

For backward compatibility, string-based learning rate schedules are also supported:

```yaml
# String-based ExponentialDecay schedule
learning_rate="ExponentialDecay(0.1, 1000, 0.96)"
```

Supported schedules:
- `ExponentialDecay`: Applies exponential decay to the learning rate
- `PiecewiseConstantDecay`: Uses a piecewise constant decay schedule
- `PolynomialDecay`: Applies a polynomial decay to the learning rate
- `InverseTimeDecay`: Applies inverse time decay to the learning rate

---

## Enhanced Dashboard UI (v0.2.6+)

Version 0.2.6 introduces a significantly improved NeuralDbg dashboard with a more aesthetic design and better visualization components (#452).

### Key UI Improvements
- **Dark Theme**: A modern, eye-friendly dark interface using Dash Bootstrap components
- **Responsive Design**: Better layout that adapts to different screen sizes
- **Improved Visualizations**: Enhanced tensor flow animations and shape propagation charts
- **Real-time Updates**: Fixed WebSocket connectivity for smoother data streaming

### Using the Enhanced Dashboard
```bash
# Start the dashboard with dark theme (default in v0.2.6+)
neural debug my_model.neural --theme dark

# Use light theme if preferred
neural debug my_model.neural --theme light
```

### Dashboard Components
- **Execution Trace**: Real-time visualization of layer activations
- **Gradient Flow**: Analysis of gradient magnitudes across layers
- **Dead Neuron Detection**: Identification of inactive neurons
- **Resource Monitoring**: CPU/GPU/Memory usage tracking
- **Shape Propagation**: Interactive visualization of tensor shapes

---

## Blog Support (v0.2.6+)

Neural DSL now includes infrastructure for blog content with markdown support and Dev.to integration (#445).

### Blog Directory Structure
```
docs/
  blog/
    README.md             # Blog overview and guidelines
    blog-list.json       # Metadata for all blog posts
    website_*.md         # Posts for the website
    devto_*.md           # Posts formatted for Dev.to
```

### Creating a Blog Post
1. Create a new markdown file in the `docs/blog/` directory
2. Use the `website_` prefix for website posts or `devto_` prefix for Dev.to posts
3. Add metadata to `blog-list.json`

### Dev.to Integration
Posts with the `devto_` prefix include special frontmatter for Dev.to:

```markdown
---
title: "Your Title Here"
published: true
description: "Brief description of your post"
tags: tag1, tag2, tag3
cover_image: https://url-to-your-cover-image.png
---

# Your Content Here
```

---

## Development Guide

### Setup & Testing
```bash
# Install dependencies
pip install -r requirements.txt
pre-commit install

# Run tests
pytest test_parser.py -v
pytest -k "transformer_validation" --log-level=DEBUG

# Generate coverage report
coverage run -m pytest && coverage html
```

### Linting Rules
- Type hint enforcement
- Parameter validation checks
- HPO syntax verification
- Device configuration validation

### Contribution Requirements
1. Unit tests for new features
2. Documentation updates
3. Backward compatibility checks
4. Pre-commit hook validation

---

## Migration Guide

### From v0.2.4 to v0.2.5

#### HPO Optimizer Improvements

**Old Style (v0.2.4):**
```yaml
# Quoted optimizer parameters
optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"

# Limited support for nested HPO in learning rate schedules
```

**New Style (v0.2.5):**
```yaml
# No quotes needed for optimizer parameters
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))

# Full support for nested HPO in learning rate schedules
optimizer: SGD(
  learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)),
    1000,
    HPO(range(0.9, 0.99, step=0.01))
  )
)
```

#### Multi-Framework Support

In v0.2.5, HPO works seamlessly across both PyTorch and TensorFlow backends:

```bash
# Run HPO with PyTorch backend
neural compile mnist_hpo.neural --backend pytorch --hpo

# Run HPO with TensorFlow backend
neural compile mnist_hpo.neural --backend tensorflow --hpo
```

### From v0.2.1 to v0.2.2

#### Old Style (Deprecated)
```yaml
network OldStyle {
    layers: Dense("64")  # String numbers
    optimizer: Adam(learning_rate="0.001")
}
```

#### New Style (Recommended)
```yaml
network NewStyle {
    layers: Dense(64)    # Integer numbers
    optimizer: Adam(learning_rate=0.001)
}
```


[Report Issues](https://github.com/Lemniscate-world/Neural/issues)

